/* (hookshot) The R button's item choice as a wheel, Twilight Princess style (the Gameplay tab's Item wheel row).

   f_802f1380 is every menu screen's update (each frame, with the screen in x0; nx2/oot_patches.py has it call
   host/oot_menu.c's hos_oot_menu_tick first, which calls hos_wheel_menu_tick here first). The R item row's screen has the vtable
   0x84c0d430 (N_SlideList_00), a persistent object that updates only while the row is shown:
   - screen+0x40 is the slide list's node; node+0x44 its cursor (an item's index: it opens on the equipped item, RIGHT
     is +1, LEFT -1, no wrapping); node+0x60 an array of 10 icon panes, the name of a pane's item behind pane+0x88.
     With N items, pane 9-i holds item i (seen with N = 3).
   The wheel (drawn by the renderer, gpu/vk/vk_menu.cpp, from hos_wheel_state) puts the items in a ring, the first at
   the top and the rest clockwise. The left stick points at one (hos_wheel_input takes it from the game while the row is
   shown, the row turns with it too); the row is turned there with D-pad taps (60 ms down, 60 up, as main's mouse wheel
   does), and R is kept held for the game until it gets there, so letting go of R early still equips the item pointed at.

   HOS_ITEM_WHEEL=0 or 1: the row's setting overridden. HOS_WHEEL_TRACE=1: the openings, the taps and R kept held, logged. */
#include "hos.h"
#include "../gpu/vk/vk_api.h"
#include "../gpu/vk/compat/hookshot_env.h"
#include "../gpu/vk/oot_cheat_items.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITEM_CHOICE_VTABLE 0x84c0d430ULL
#define ROW_PANES 10
#define WHEEL_PI 3.14159265358979323846

enum { B_A = 0, B_B, B_X, B_Y, B_LS, B_RS, B_L, B_R, B_ZL, B_ZR, B_PLUS, B_MINUS, B_LEFT, B_UP, B_RIGHT, B_DOWN };
/* the row's buttons: R, and L, which opens the same row ("Quick Select Items" on both): "R" below means either */
#define HOLD_BUTTONS ((1ULL << B_R) | (1ULL << B_L))

/* a guest read inside the image or the heap (a stray pointer elsewhere is not followed). Compared as distances, so a
   pointer near the top of the space (0xffff...) cannot wrap around into range */
static int valid(uint64_t a, size_t n) {
    if (a >= hos_image_base && a < hos_image_end && n <= hos_image_end - a) return 1;
    const uint64_t lo = HOS_HEAP_BASE, hi = HOS_HEAP_BASE + (32ULL << 30);
    return a >= lo && a < hi && n <= hi - a;
}
static int peek(uint64_t a, void *dst, size_t n) {
    if (!valid(a, n)) return 0;
    memcpy(dst, GP(a), n);
    return 1;
}
static uint64_t peek64(uint64_t a) {
    uint64_t v = 0;
    return peek(a, &v, 8) ? v : 0;
}
/* an identifier at a: letters, digits and _ only, 4 to 63 of them, ended by a NUL or a non-identifier byte */
static int ident_at(uint64_t a, char *buf, size_t cap) {
    if (!peek(a, buf, cap - 1)) return 0;
    buf[cap - 1] = 0;
    size_t i = 0;
    for (; buf[i]; i++) {
        const unsigned char ch = (unsigned char)buf[i];
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_')) break;
    }
    buf[i] = 0;
    return i >= 4 && i < cap - 1;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the wheel                                                                                                         */

/* what the player reads for a game name: the debug inventory's labels (gpu/vk/oot_cheat_items.h), else the name
   without its kind ("Item_Bottle_Fairy" -> "Bottle Fairy") */
#define WHEEL_SECTION(title)
#define WHEEL_ITEM(label, name, kind) {name, label},
static const struct { const char *name, *label; } item_labels[] = {
    OOT_CHEAT_ITEMS(WHEEL_SECTION, WHEEL_ITEM)
};
static void label_of(const char *name, char *out, size_t cap) {
    for (size_t i = 0; i < sizeof item_labels / sizeof item_labels[0]; i++)
        if (!strcmp(item_labels[i].name, name)) {
            snprintf(out, cap, "%s", item_labels[i].label);
            return;
        }
    const char *s = name;
    static const char *const kinds[] = {"Item_", "Weapon_Range_", "Weapon_", "Armor_Child_Mask_", "Armor_Adult_", "Armor_"};
    for (size_t i = 0; i < sizeof kinds / sizeof kinds[0]; i++)
        if (!strncmp(s, kinds[i], strlen(kinds[i]))) {
            s += strlen(kinds[i]);
            break;
        }
    size_t n = 0;
    for (; *s && n + 1 < cap; s++) out[n++] = *s == '_' ? ' ' : *s;
    out[n] = 0;
}

/* shared with the input thread (hos_wheel_input) */
static volatile int64_t row_tick_ns;    /* the row's last update */
static volatile int64_t other_tick_ns;  /* any other screen's last update (the HUD's, a text box's, the pause menu's) */
static volatile int row_count;          /* its items */
static volatile int row_cursor = -1;    /* its cursor */
static volatile int wheel_target = -1;  /* the item the stick pointed at last this opening, -1 none yet */
static volatile int wheel_pick = -1;    /* the item the stick points at now, -1 at rest */
static volatile int wheel_opening;      /* counts the openings (the input side starts afresh on a new one) */
static volatile uint32_t row_known;     /* the items whose names have been seen (bit i: item i, the first 32) */
static volatile int wheel_home = -1;    /* the cursor the row opened on, while the survey turns it (-1 none) */

/* The count the game shows on an item (the HUD's number): the pouch's own count of what it uses, as host/hos_main.c's
   inventory refresh publishes it twice a second (rx_option(OOT_CHEAT_ITEM_BASE + its place in OOT_CHEAT_ITEMS)); -1 for
   an item without one, or before the first refresh */
static int amount_of(const char *name) {
    int rx_option(int id, int otherwise);
    static const struct { const char *item, *counted; } counted[] = {
        {"Item_DekuStick", "Item_DekuStick"}, {"Item_DekuNut", "Item_DekuNut"}, {"Item_Bomb", "Item_Bomb"},
        {"Item_Bombchu", "Item_Bombchu"}, {"Item_MagicBean", "Item_MagicBean"},
        {"Weapon_Range_SlingShot", "Item_DekuSeed"}, {"Weapon_Range_Bow", "Item_Arrow_Normal"},
        {"Item_Arrow_Fire", "Item_Arrow_Normal"}, {"Item_Arrow_Ice", "Item_Arrow_Normal"},
        {"Item_Arrow_Light", "Item_Arrow_Normal"},
    };
    for (size_t k = 0; k < sizeof counted / sizeof counted[0]; k++) {
        if (strcmp(counted[k].item, name)) continue;
        for (size_t i = 0; i < sizeof item_labels / sizeof item_labels[0]; i++)
            if (!strcmp(item_labels[i].name, counted[k].counted)) {
                const int n = rx_option(OOT_CHEAT_ITEM_BASE + (int)i, -1);
                return n >= 0 && n < 1000 ? n : -1;
            }
    }
    return -1;
}

#define ROW_OPEN_NS 100000000LL    /* the row counts as shown until 100 ms after its last update */
#define ROW_STALL_NS 500000000LL   /* ... or 500 ms while R is held (a game stall is not a close) */
#define ROW_GONE_NS 60000000LL     /* ... unless other screens went on updating 60 ms without it: closed (row_gone) */

/* The row is gone though its last update is recent: the game went on (its other screens updated) for ROW_GONE_NS without
   it, so the game closed it (a text box, a cutscene, damage), not a stall (where nothing updates). The synthetic R and
   D-pad taps stop then, rather than after ROW_STALL_NS, so they never reach play for more than a frame or two */
static int row_gone(int64_t last) {
    return last && __atomic_load_n(&other_tick_ns, __ATOMIC_ACQUIRE) - last > ROW_GONE_NS;
}

static int wheel_trace(void) {   /* HOS_WHEEL_TRACE=1: the openings, the taps, R kept held */
    static int on = -1;
    if (on < 0) {
        on = env_on("HOS_WHEEL_TRACE");
    }
    return on;
}

/* the state the renderer draws: a seqlock (odd while written) */
static MkwVkWheel wheel_state;
static volatile int state_lock;   /* (the menu ticks and hos_wheel_state's close, one at a time) */

static void publish(const MkwVkWheel *next) {
    if (!memcmp((const char *)next + sizeof next->seq, (const char *)&wheel_state + sizeof wheel_state.seq,
                sizeof *next - sizeof next->seq))
        return;
    __atomic_add_fetch(&wheel_state.seq, 1, __ATOMIC_ACQ_REL);
    memcpy((char *)&wheel_state + sizeof wheel_state.seq, (const char *)next + sizeof next->seq, sizeof *next - sizeof next->seq);
    __atomic_add_fetch(&wheel_state.seq, 1, __ATOMIC_ACQ_REL);
}

const MkwVkWheel *hos_wheel_state(void) {
    /* (the row stopped updating: closed. Its own update cannot say so, so the renderer's call does) */
    const int64_t last = __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE);
    if (wheel_state.open && (hos_now_ns() - last > ROW_OPEN_NS || row_gone(last)) &&
        !__atomic_exchange_n(&state_lock, 1, __ATOMIC_ACQUIRE)) {
        MkwVkWheel next = wheel_state;
        next.open = 0, next.pick = -1;
        publish(&next);
        __atomic_store_n(&state_lock, 0, __ATOMIC_RELEASE);
    }
    return &wheel_state;
}

/* The row's list. Its length is the slide list's data source's (node+0x68, a count at +0x78). Only part of it is loaded
   at a time, about 4 items behind the cursor to 5 ahead, into 10 recycled panes (node+0x60, as many as node+0x58 says):
   a pane names its item at +0x88 and its layout part at +0x20, and the slot records at node+0xc0 (0x40 bytes each)
   pair a part (+0x18, the pane's + 0x40) with the data index it shows (+0x28). The panes' own order says nothing (they
   are a ring that turns as the row scrolls: reading them in order equipped the wrong item). The items seen loaded are
   kept by index, so the wheel shows the whole row once each part of it has been loaded; a change of length (an item
   gained or used up) forgets them. */
static char row_names[MKWVK_WHEEL_MAX][48];
static int row_names_count = -1;

static int row_length(uint64_t node) {
    int32_t n = -1;
    const uint64_t source = peek64(node + 0x68);
    if (!source || !peek(source + 0x78, &n, 4) || n < 0 || n > 4096) return -1;
    return n;
}
static void row_learn(uint64_t node, int count) {
    if (count != row_names_count) memset(row_names, 0, sizeof row_names), row_names_count = count;
    uint64_t rec_part[16];
    int32_t rec_index[16];
    for (int k = 0; k < 16; k++) {
        rec_part[k] = 0, rec_index[k] = -1;
        if (k < ROW_PANES) {
            rec_part[k] = peek64(node + 0xc0 + 0x40ULL * (uint64_t)k + 0x18);
            peek(node + 0xc0 + 0x40ULL * (uint64_t)k + 0x28, &rec_index[k], 4);
        }
    }
    int32_t npanes = ROW_PANES;
    peek(node + 0x58, &npanes, 4);
    if (npanes < 0 || npanes > 16) npanes = ROW_PANES;
    const uint64_t panes = peek64(node + 0x60);
    int seen_idx[16], nseen = 0, changed = 0;
    char seen_name[16][48];
    for (int i = 0; panes && i < npanes && nseen < 16; i++) {
        const uint64_t pane = peek64(panes + 8ULL * (uint64_t)i);
        char name[64];
        if (!valid(pane, 0x100) || !ident_at(peek64(pane + 0x88), name, sizeof name)) continue;
        const uint64_t part = peek64(pane + 0x20) + 0x40;
        for (int k = 0; k < ROW_PANES; k++)
            if (rec_part[k] == part) {
                const int idx = rec_index[k];
                if (idx >= 0 && idx < count && idx < MKWVK_WHEEL_MAX) {
                    seen_idx[nseen] = idx;
                    snprintf(seen_name[nseen], sizeof seen_name[nseen], "%s", name);
                    if (row_names[idx][0] && strcmp(row_names[idx], seen_name[nseen])) changed = 1;
                    nseen++;
                }
                break;
            }
    }
    /* a loaded item that is not the one kept for its place: the row changed with its length the same (a bottle filled
       or emptied, one item traded for another), so the others kept may be stale too: they are learned afresh (never
       seen while the row only scrolls) */
    if (changed) {
        if (wheel_trace()) hos_log("wheel: the row's items changed (same length %d): learning them again", count);
        memset(row_names, 0, sizeof row_names);
    }
    for (int s = 0; s < nseen; s++) {
        const int idx = seen_idx[s];
        if (!strcmp(row_names[idx], seen_name[s])) continue;
        if (wheel_trace()) hos_log("wheel: item %d is %s", idx, seen_name[s]);
        snprintf(row_names[idx], sizeof row_names[idx], "%s", seen_name[s]);
    }
}

static void wheel_tick(uint64_t screen, int opened) {
    const uint64_t node = peek64(screen + 0x40);
    int32_t cursor = -1;
    if (node) peek(node + 0x44, &cursor, 4);
    MkwVkWheel next;
    memset(&next, 0, sizeof next);
    next.seq = 0;
    next.open = 1;
    const int length = node ? row_length(node) : -1;
    if (length >= 0) {
        row_learn(node, length);
        next.count = length < MKWVK_WHEEL_MAX ? length : MKWVK_WHEEL_MAX;
    }
    uint32_t known = 0;
    for (int i = 0; i < MKWVK_WHEEL_MAX; i++) {
        next.amount[i] = -1;
        if (i >= next.count) continue;
        if (row_names[i][0]) {
            known |= 1u << i;
            snprintf(next.name[i], sizeof next.name[i], "%s", row_names[i]);
            label_of(row_names[i], next.label[i], sizeof next.label[i]);
            next.amount[i] = amount_of(row_names[i]);
        } else {
            snprintf(next.label[i], sizeof next.label[i], "?");   /* (not loaded yet: the survey is on its way there) */
        }
    }
    __atomic_store_n(&row_known, known, __ATOMIC_RELEASE);
    /* (while the survey turns the row and nothing is picked, the wheel keeps showing the item it opened on: that is what
       letting go of R equips) */
    const int home = __atomic_load_n(&wheel_home, __ATOMIC_ACQUIRE);
    next.cursor = home >= 0 && home < next.count && __atomic_load_n(&wheel_target, __ATOMIC_ACQUIRE) < 0 ? home : cursor;
    next.pick = __atomic_load_n(&wheel_pick, __ATOMIC_ACQUIRE);
    if (next.pick >= next.count) next.pick = -1;
    __atomic_store_n(&row_count, next.count, __ATOMIC_RELEASE);
    __atomic_store_n(&row_cursor, cursor, __ATOMIC_RELEASE);
    static int ticks_open;   /* (the items once an opening, 10 updates in: the row fills its panes over its first frames) */
    if (opened) {
        __atomic_add_fetch(&wheel_opening, 1, __ATOMIC_ACQ_REL);
        ticks_open = 0;
    }
    if (++ticks_open == 10) {
        char list[MKWVK_WHEEL_MAX * 34] = "";
        size_t n = 0;
        for (int i = 0; i < next.count && n < sizeof list - 40; i++)
            n += (size_t)snprintf(list + n, sizeof list - n, "%s%s%s", i ? ", " : "", i == cursor ? "*" : "", next.label[i]);
        hos_log("wheel: the row (screen %#llx) shows %d items: %s", (unsigned long long)screen, next.count, list);
    }
    publish(&next);
}

/* the slot in direction (x, y) (y up): item 0 at the top, the rest clockwise, each a 360/count wedge round its own
   direction */
static int slot_toward(double x, double y, int count) {
    double a = atan2(x, y);   /* (0 up, clockwise positive) */
    if (a < 0) a += 2 * WHEEL_PI;
    return (int)floor(a / (2 * WHEEL_PI / count) + 0.5) % count;
}

/* the slot under the mouse pointer (client pixels px, py in a w x h client), -1 in the wheel's hub. The wheel is centred
   on the window, its unit a hundredth of the height and its hub 12 units across (gpu/vk/vk_menu.cpp wheel_geometry) */
static int slot_under(int32_t px, int32_t py, int32_t w, int32_t h, int count) {
    if (count <= 0 || w <= 0 || h <= 0) return -1;
    const double u = h / 100 > 1 ? h / 100 : 1, dx = px - w / 2, dy = py - h / 2;
    if (dx * dx + dy * dy <= 12.0 * u * 12.0 * u) return -1;
    return slot_toward(dx, -dy, count);
}

/* The survey: the row loads only about 4 items behind its cursor to 5 ahead, so a wheel opened on part of it has slots it
   cannot name yet. Hidden behind the wheel's backdrop, the row is turned (the same taps) to the nearest stop that loads
   an unseen item, then back to the item it opened on (home), while the player has picked nothing and holds R; a pick
   takes over at once, and R let go first goes home before the game sees it. 3 s at most an opening. The stop for the nearest unseen item (a margin of one inside the window), the cursor itself while one is in the
   window but not loaded yet, -1 when all are seen */
static int survey_stop(int cursor, int count, uint32_t known) {
    int best = -1, best_d = 1 << 30;
    for (int i = 0; i < count && i < 32; i++) {
        if (known >> i & 1) continue;
        int stop = i > cursor + 4 ? i - 4 : i < cursor - 3 ? i + 3 : cursor;
        if (stop > count - 1) stop = count - 1;
        if (stop < 0) stop = 0;
        const int d = abs(stop - cursor);
        if (d < best_d) best_d = d, best = stop;
    }
    return best;
}

/* The game's controller state, before the game sees it (nn_hid.c npad_states). While the row is shown: the left stick
   is the wheel's (the game's row would turn with it), the D-pad gets the taps that turn the row to the item the stick
   pointed at, and R stays held until the row is there (at most 4 s after the player let go, and only while the row is
   shown: the game closing it ends the R and the taps at once, row_gone).
   The mouse (host/display.c): while the wheel is open it is let go and its pointer shown. Moved, the item under it is
   the pick as the stick's is (until the stick is pushed again); a left click on an item picks it and closes the wheel
   (R let go for the game however it is held, and kept from it until the player lets go of it too). */
/* The sounds (nn_audio.c): the game's row plays its own open, select and close; its scroll tick, once a tap of the hidden
   row, is silent while the row updates (and 0.4 s after), and the wheel ticks once each new slot it points at instead. */
int hos_wheel_hush(void) {
    const int64_t last = __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE);
    return last && hos_now_ns() - last < 400000000LL;
}

uint64_t hos_wheel_input(uint64_t buttons, int32_t stick[4]) {
    void hos_wheel_mouse_open(int open);   /* display.c */
    void hos_audio_cue(const char *name);  /* nn_audio.c */
    int hos_wheel_pointer(int32_t *x, int32_t *y, int32_t *w, int32_t *h, int *clicked, int32_t *click_x, int32_t *click_y);
    static int opening_seen = -1;
    static int target = -1;
    static int64_t tap_down_until, tap_free_at, r_let_go_at;
    static int tap_dir, r_kept;
    static int was_open, click_close, r_block, mouse_live;
    static int32_t ref_x, ref_y;
    /* (home: the row's cursor as the opening settles, dropped for good at the first pick. It follows the cursor for the
       opening's first 200 ms, and the survey waits as long: after the row's items changed, its first update still gave
       the old cursor (8 for adult Link's Light Arrow, then 1 for the child's Slingshot), and the survey took the row
       back to the old place and equipped what was there now) */
    static int home = -1, home_taken;
    static int64_t survey_until, opened_at;
    const int64_t now = hos_now_ns();
    const int64_t last = __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE);
    /* (shown: updated within 100 ms; while R is held, the game's or ours, within 500 ms: the game stalled for 157 ms
       while the row was open, and taking that for a close dropped the target, so a let-go R equipped whatever the row
       was passing) */
    const int open = last && !row_gone(last) && (now - last < ROW_OPEN_NS || (r_kept && now - last < ROW_STALL_NS));
    const int opening = __atomic_load_n(&wheel_opening, __ATOMIC_ACQUIRE);
    if (open != was_open) {
        if (!open && wheel_trace())
            hos_log("wheel: the row closes (%s; R %s)", row_gone(last) ? "the game went on without it" : "no update",
                    r_kept ? "kept held: let go now" : "the player's");
        was_open = open, hos_wheel_mouse_open(open);
    }
    int32_t mx = 0, my = 0, mw = 0, mh = 0, cx = 0, cy = 0;
    int clicked = 0;
    const int mouse_ok = hos_wheel_pointer(&mx, &my, &mw, &mh, &clicked, &cx, &cy);
    if (!open) {
        if (target >= 0 || tap_dir) target = -1, tap_dir = 0, __atomic_store_n(&wheel_target, -1, __ATOMIC_RELEASE);
        __atomic_store_n(&wheel_pick, -1, __ATOMIC_RELEASE);
        r_let_go_at = 0, r_kept = 0;
        if (home >= 0) home = -1, __atomic_store_n(&wheel_home, -1, __ATOMIC_RELEASE);
        /* (closed by a click: R kept from the game until the player lets go of it, or the row would open again) */
        if (click_close) r_block = 1, click_close = 0;
        if (r_block) {
            if (buttons & HOLD_BUTTONS) buttons &= ~HOLD_BUTTONS;
            else r_block = 0;
        }
        return buttons;
    }
    if (opening != opening_seen) {
        opening_seen = opening, target = -1, tap_dir = 0, tap_free_at = 0, r_let_go_at = 0;
        click_close = 0, mouse_live = 0, ref_x = mx, ref_y = my;
        home = -1, home_taken = 0, survey_until = now + 3000000000LL, opened_at = now;
        __atomic_store_n(&wheel_home, -1, __ATOMIC_RELEASE);
    }
    const int count = __atomic_load_n(&row_count, __ATOMIC_ACQUIRE);
    const int cursor = __atomic_load_n(&row_cursor, __ATOMIC_ACQUIRE);
    const int r_pressed = (buttons & HOLD_BUTTONS) != 0;   /* (the player's own R or L) */
    const int settled_in = now - opened_at >= 200000000LL;   /* (the opening's first updates are over) */
    if (!home_taken && cursor >= 0 && count > 0) {
        if (home != cursor) home = cursor, __atomic_store_n(&wheel_home, home, __ATOMIC_RELEASE);
        if (settled_in) home_taken = 1;
    }
    const double x = stick[0], y = stick[1];
    int pick = -1;
    if (count > 0 && x * x + y * y > 16000.0 * 16000.0) pick = slot_toward(x, y, count);
    stick[0] = stick[1] = 0;
    /* the mouse's pick: once the pointer has moved this opening, and since the stick last pointed (a pointer left on an
       item does not take the pick back from the stick) */
    if (pick >= 0) {
        ref_x = mx, ref_y = my, mouse_live = 0;
    } else if (mouse_ok) {
        if (!mouse_live && (abs(mx - ref_x) > 4 || abs(my - ref_y) > 4)) mouse_live = 1;
        if (mouse_live) pick = slot_under(mx, my, mw, mh, count);
    }
    if (clicked && !click_close) {
        const int at = slot_under(cx, cy, mw, mh, count);
        if (at >= 0) {
            pick = at, click_close = 1, mouse_live = 1;
            if (wheel_trace()) hos_log("wheel: clicked item %d (cursor %d): closing", at, cursor);
        }
    }
    __atomic_store_n(&wheel_pick, pick, __ATOMIC_RELEASE);
    if (pick >= 0 && pick != target && (!click_close || clicked)) {   /* (after a click, the clicked item stays the target) */
        const int shown = target >= 0 ? target : home >= 0 ? home : cursor;   /* (the slot it pointed at until now) */
        if (pick != shown) hos_audio_cue("SE_Sys_ShortCut_CursolMove_00");
        target = pick;
        __atomic_store_n(&wheel_target, target, __ATOMIC_RELEASE);
        /* (a pick ends the survey and home with it: once the pick is equipped and cleared, nothing turns the row back) */
        if (home >= 0) home = -1, __atomic_store_n(&wheel_home, -1, __ATOMIC_RELEASE);
        home_taken = 1;
    }
    /* R held for the game until the row has reached the target and the last tap has been let go: a D-pad press that
       outlives the row reaches play, where LEFT opened the pause menu (the cursor moves on a tap's press, so the
       target was reached with LEFT still down) */
    if (tap_dir && now >= tap_down_until) tap_dir = 0;
    /* where the taps go: the pick; with none, the survey's next stop while it runs, else home */
    int aim = target;
    if (target < 0 && home >= 0) {
        const int stop = settled_in && r_pressed && !click_close && now < survey_until
                             ? survey_stop(cursor, count, __atomic_load_n(&row_known, __ATOMIC_ACQUIRE))
                             : -1;
        aim = stop >= 0 ? stop : home;
    }
    const int settled = (target >= 0 ? cursor == target : home < 0 || cursor == home) && !tap_dir;
    if (click_close) buttons &= ~HOLD_BUTTONS;   /* (a click closes: R let go for the game, as by the player) */
    int r_held = (buttons & HOLD_BUTTONS) != 0;
    if (!r_held) {
        if (!r_let_go_at) {
            r_let_go_at = now;
            if (!settled && wheel_trace()) hos_log("wheel: R let go before the row got there (cursor %d, target %d): kept held", cursor, target);
        }
        /* (at most 4 s: a 20-item row's far end is 19 taps of 0.12 s away) */
        if (!settled && now - r_let_go_at < 4000000000LL) r_held = 1, buttons |= 1ULL << B_R;
    } else {
        r_let_go_at = 0;
    }
    r_kept = r_held;
    if (!r_held && settled && target >= 0) {   /* (done: the next opening starts afresh, however soon it comes) */
        target = -1;
        __atomic_store_n(&wheel_target, -1, __ATOMIC_RELEASE);
    }
    /* a tap toward the target, only while R is held, each after the last was let go and the cursor had a frame to follow */
    if (!r_held) tap_dir = 0;
    if (r_held && !tap_dir && aim >= 0 && cursor >= 0 && cursor != aim && now >= tap_free_at) {
        tap_dir = aim > cursor ? 1 : -1;
        tap_down_until = now + 60000000LL, tap_free_at = now + 120000000LL;
        if (wheel_trace())
            hos_log("wheel: tap %s (cursor %d, %s %d)", tap_dir > 0 ? "RIGHT" : "LEFT", cursor,
                    target >= 0 ? "target" : aim == home ? "home" : "survey", aim);
    }
    if (tap_dir) buttons |= 1ULL << (tap_dir > 0 ? B_RIGHT : B_LEFT);
    return buttons;
}

/* ---------------------------------------------------------------------------------------------------------------- */

/* The Gameplay tab's Item wheel row (off by default; HOS_ITEM_WHEEL=0 or 1 overrides it). Off, the row's updates are not
   taken: the wheel stays closed, so the controls, the row's sounds and the picture are the game's own. Read each update,
   so a change applies at once (an open wheel closes with the row's next update) */
static int wheel_enabled(void) {
    int rx_option(int id, int otherwise);
    static int forced = -2;
    if (forced == -2) {
        const char *e = getenv("HOS_ITEM_WHEEL");
        forced = e ? env_on("HOS_ITEM_WHEEL") : -1;   /* (set: it decides; unset: the menu's row) */
    }
    return forced >= 0 ? forced : rx_option(HOS_OPT_ITEM_WHEEL, 0) != 0;
}

/* (host/oot_menu.c's hos_oot_menu_tick, first) a menu screen's update; 1 when it is the item row and the wheel has it: the
   game's menus' mouse leaves that screen alone then (with the wheel off, the row is the game's, its mouse too) */
int hos_wheel_menu_tick(Ctx *c, uint64_t screen_arg) {
    (void)c;
    const uint64_t screen = screen_arg;
    const uint64_t vt = peek64(screen);
    const int mine = vt == ITEM_CHOICE_VTABLE && wheel_enabled();
    if (vt != ITEM_CHOICE_VTABLE) __atomic_store_n(&other_tick_ns, hos_now_ns(), __ATOMIC_RELEASE);   /* (row_gone) */
    /* (the screens update on several worker threads at once: one at a time here, the others skip this frame) */
    if (__atomic_exchange_n(&state_lock, 1, __ATOMIC_ACQUIRE)) return mine;
    if (mine) {
        static uint64_t open_screen;
        const int64_t now = hos_now_ns();
        const int64_t last = __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE);
        const int opened = screen != open_screen || !last || now - last > ROW_STALL_NS;
        if (opened && wheel_trace())
            hos_log("wheel: the row opens (screen %#llx%s, %.0f ms after its last update)", (unsigned long long)screen,
                    screen != open_screen ? ", a new one" : "", last ? (double)(now - last) / 1e6 : -1.0);
        open_screen = screen;
        __atomic_store_n(&row_tick_ns, now, __ATOMIC_RELEASE);
        wheel_tick(screen, opened);
    }
    __atomic_store_n(&state_lock, 0, __ATOMIC_RELEASE);
    return mine;
}
