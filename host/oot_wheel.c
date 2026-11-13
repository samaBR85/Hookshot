/* (hookshot) The R button's item choice as a wheel, Twilight Princess style (the Gameplay tab's Item wheel row).

   f_802f1380 is every menu screen's update (each frame, with the screen in x0; nx2/oot_patches.py has it call
   host/oot_menu.c's hos_oot_menu_tick first, which calls hos_wheel_menu_tick here first). The R item row's screen has the vtable
   0x84c0d430 (N_SlideList_00), a persistent object that updates only while the row is shown:
   - screen+0x40 is the slide list's node; node+0x44 its cursor (an item's index: it opens on the equipped item, RIGHT
     is +1, LEFT -1, no wrapping); node+0x60 an array of 10 icon panes, the name of a pane's item behind pane+0x88.
     With N items, pane 9-i holds item i (seen with N = 3).
   The wheel (drawn by the renderer, gpu/vk/vk_menu.cpp, from hos_wheel_state) shows the items on pages of 8 fixed slots,
   45 degrees apart (what a stick points at surely), in an order of its own (wheel_layout: the game's row reorders itself
   at every use, the wheel does not). Either stick points at one, the D-pad's Left and Right or ZL and ZR
   the right stick turn the page (hos_wheel_input takes them from the game while the row is shown); the hidden row is
   turned to the item chosen with D-pad taps (60 ms down, 60 up, as main's mouse wheel does), and R is kept held for the
   game until it gets there, so letting go of R early still equips the item pointed at.

   HOS_ITEM_WHEEL=0 or 1: the row's setting overridden. HOS_WHEEL_TRACE=1: the openings, the taps and R kept held, logged. */
#include "hos.h"
#include "hos_paths.h"   /* (the game folder: wheel_layout.txt) */
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
static volatile int wheel_pick = -1;    /* the slot the stick (or the mouse) points at now, on the page shown; -1 none */
static volatile int wheel_opening;      /* counts the openings (the input side starts afresh on a new one) */
static volatile uint32_t row_known;     /* the row's places whose item is known (bit i: place i, the first 32) */
static volatile int wheel_home = -1;    /* the cursor the row opened on, while the survey turns it (-1 none) */
/* the pages: the input thread turns them (wheel_page, kept from one opening to the next), the menu tick lays them out */
static volatile int wheel_page;         /* the page shown */
static volatile int page_count;         /* the pages */
static volatile uint32_t page_mask;     /* the page laid out (above bit 8) and its slots that hold an item (bit s: slot s) */
/* moving an item (Y, wheel_layout.txt): the input thread holds one (move_from: page * 8 + slot as shown, -1 none) and puts it
   down (drop_req: from << 16 | to, and a new drop_seq); the menu tick moves it in the layout */
static volatile int move_from = -1;
static volatile int wheel_closed;       /* (the input side saw the row close: the menu tick's next update is a new opening) */
static volatile int drop_req, drop_seq;
/* the item chosen: the input thread asks (choice_key, a new choice_seq), the menu tick answers (choice_ack = that seq) */
static volatile int choice_key = -1;    /* page * MKWVK_WHEEL_SLOTS + slot of the item chosen this opening, -1 none */
static volatile int choice_seq;         /* counts the choices */
static volatile int choice_ack = -1;    /* the choice the answers below are for */
static volatile int choice_row = -1;    /* the row's place to turn to for it (the next stop of a search while unseen) */
static volatile int choice_here;        /* 1: the row's cursor is on the chosen item */
static volatile int choice_lost;        /* 1: the whole row is known and the item is not in it (nothing to turn to) */
static volatile int choice_item = -1;   /* its number in the game (PouchMgr's AssignableItem), -1 not found: then the row is turned */
static volatile int equip_item = -1, equip_seq;   /* (the input side: an item to put on ZR at the HUD's next update) */
static char equip_name[48];             /* (the menu tick's: the chosen item's name, for the row's places after it is equipped) */
static int cy_item_of_name(Ctx *c, const char *name);   /* (below, with Cycle Items) */
static int cy_zr_name(Ctx *c, char *out, size_t cap);    /* (below: the name of the item on ZR now) */

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
        {"Item_Arrow_Light", "Item_Arrow_Normal"}, {"Item_Arrow_Normal", "Item_Arrow_Normal"},
        {"Pouch_Range_BowAndArrow_Normal", "Item_Arrow_Normal"}, {"Pouch_Range_BowAndArrow_Fire", "Item_Arrow_Normal"},
        {"Pouch_Range_BowAndArrow_Ice", "Item_Arrow_Normal"}, {"Pouch_Range_BowAndArrow_Light", "Item_Arrow_Normal"},
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
   are a ring that turns as the row scrolls: reading them in order equipped the wrong item).

   The row is in the order of use: the item equipped last is first, and equipping the item at place k moves it to 0 and
   those before it one on (the log's row lines: "*Empty Bottle, Bombchus, Fairy Bow, ..." after Empty Bottle was taken
   from place 7). So the places seen are kept (row_names) and turned that way at each opening (row_to_front) instead of
   learned again, and the items the wheel shows are kept apart (item_names, a set) and laid out in an order of their own
   (wheel_layout): a slot keeps its item from one opening to the next, whatever was used last. A place seen holding
   another item takes it (a bottle filled, an item gained); once every place is known, the set is the row's. */
static char row_names[MKWVK_WHEEL_MAX][48];   /* by place in the row, "" where not known */
static int row_names_count = -1;
static char item_names[MKWVK_WHEEL_MAX][48];  /* the items known to be in the row, in no order */
static int item_n;

static int row_length(uint64_t node) {
    int32_t n = -1;
    const uint64_t source = peek64(node + 0x68);
    if (!source || !peek(source + 0x78, &n, 4) || n < 0 || n > 4096) return -1;
    return n;
}
static int row_find(const char *name) {
    for (int i = 0; i < MKWVK_WHEEL_MAX && i < row_names_count; i++)
        if (row_names[i][0] && !strcmp(row_names[i], name)) return i;
    return -1;
}
static void items_add(const char *name) {
    for (int i = 0; i < item_n; i++)
        if (!strcmp(item_names[i], name)) return;
    if (item_n < MKWVK_WHEEL_MAX) snprintf(item_names[item_n++], sizeof item_names[0], "%s", name);
}
/* the game's own reordering at a close: the item at place k (equipped) first, those before it one on */
static void row_to_front(int k) {
    if (k <= 0 || k >= MKWVK_WHEEL_MAX || k >= row_names_count) return;
    char moved[48];
    memcpy(moved, row_names[k], sizeof moved);
    memmove(row_names[1], row_names[0], (size_t)k * sizeof row_names[0]);
    memcpy(row_names[0], moved, sizeof moved);
}
/* the one name an item is kept under: the bow and its arrows take other names with the arrow on the bow ("Item_Arrow_Normal"
   the bow with plain arrows, "Pouch_Range_BowAndArrow_Ice" the Ice Arrow while it is the one on it, seen at the Gerudo
   Fortress), and they change as the hidden row turns past them: kept as one, the wheel's icons and slots do not change */
static void canonical(char *name, size_t cap) {
    if (!strcmp(name, "Item_Arrow_Normal") || !strcmp(name, "Pouch_Range_BowAndArrow_Normal")) snprintf(name, cap, "Weapon_Range_Bow");
    else if (!strncmp(name, "Pouch_Range_BowAndArrow_", 24)) {
        char kind[40];
        snprintf(kind, sizeof kind, "%s", name + 24);
        snprintf(name, cap, "Item_Arrow_%s", kind);
    }
}
static void row_learn(uint64_t node, int count) {
    if (count != row_names_count) {   /* (an item gained or used up: the places are learned afresh, the set is kept) */
        if (wheel_trace() && row_names_count >= 0) hos_log("wheel: the row's length went %d -> %d", row_names_count, count);
        memset(row_names, 0, sizeof row_names), row_names_count = count;
    }
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
    for (int i = 0; panes && i < npanes; i++) {
        const uint64_t pane = peek64(panes + 8ULL * (uint64_t)i);
        char name[64];
        if (!valid(pane, 0x100) || !ident_at(peek64(pane + 0x88), name, sizeof name)) continue;
        canonical(name, sizeof name);
        const uint64_t part = peek64(pane + 0x20) + 0x40;
        for (int k = 0; k < ROW_PANES; k++)
            if (rec_part[k] == part) {
                const int idx = rec_index[k];
                if (idx >= 0 && idx < count && idx < MKWVK_WHEEL_MAX && strcmp(row_names[idx], name)) {
                    for (int j = 0; j < MKWVK_WHEEL_MAX; j++)   /* (one place per item: an older place of it is stale) */
                        if (j != idx && !strcmp(row_names[j], name)) row_names[j][0] = 0;
                    if (wheel_trace()) hos_log("wheel: place %d is %s (was %s)", idx, name, row_names[idx][0] ? row_names[idx] : "not known");
                    snprintf(row_names[idx], sizeof row_names[idx], "%s", name);
                }
                if (idx >= 0 && idx < count) items_add(name);
                break;
            }
    }
    /* every place known: the set is the row (an item used up or emptied leaves it) */
    int all = count > 0 && count <= MKWVK_WHEEL_MAX;
    for (int i = 0; all && i < count; i++) all = row_names[i][0] != 0;
    if (all && item_n != count) {
        if (wheel_trace()) hos_log("wheel: the whole row known: %d items (the wheel had %d)", count, item_n);
        item_n = 0;
        for (int i = 0; i < count; i++) items_add(row_names[i]);
    }
}

/* ---- the pages ----------------------------------------------------------------------------------------------------
   The player's, one set for the child and one for the adult (rx_option(OOT_CHEAT_AGE_ID), host/hos_main.c): pages of 8
   slots, each slot an item's key (key_of: any bottle is "Item_Bottle*", the Longshot the Hookshot's slot) or empty. An item
   keeps its slot whether it is had or not: a slot whose item is not had shows empty. The items no slot names are laid out
   after them, on pages of their own ("others", in item_order's order). The pages are kept in wheel_layout.txt in the game
   folder, written again at each move (hold R, Y on an item, Y on the slot to put it in: the two swap; on the others' pages
   it leaves its slot; while one is held a new page follows the others). */
#define LAYOUT_PAGES 8
#define PAGES_MAX (LAYOUT_PAGES + MKWVK_WHEEL_MAX / MKWVK_WHEEL_SLOTS + 1)
#define LAYOUT_MAX (PAGES_MAX * MKWVK_WHEEL_SLOTS)
enum { PAGE_LAYOUT, PAGE_OTHERS, PAGE_NEW };
typedef struct {
    char slot[LAYOUT_PAGES][MKWVK_WHEEL_SLOTS][40];
    int pages;
} AgeLayout;
static AgeLayout layouts[2];   /* [0] the child's, [1] the adult's */
typedef struct {
    int kind, page;   /* PAGE_*; the layout's page (PAGE_LAYOUT, PAGE_NEW) */
} PageInfo;

/* the others' order, and the bottles' among themselves: groups (weapons, tools and magic, bottles, masks, the rest), each
   in a fixed order (an entry ending in * a prefix) */
static const char *const item_order[] = {
    "Item_DekuStick", "Item_DekuNut", "Item_Bomb", "Item_Bombchu", "Weapon_Range_SlingShot", "Weapon_Range_Bow",
    "Item_Arrow_Fire", "Item_Arrow_Ice", "Item_Arrow_Light", "Item_Boomerang", NULL,
    "Item_Hookshot", "Item_Longshot", "Item_Hammer", "Item_GlassTruth", "Item_DinFire", "Item_FaroreWind", "Item_NayruLove",
    "Item_MagicBean", NULL,
    /* (the bottles by what they hold, as the game's inventory lists them: it is not known which bottle is which) */
    "Item_Bottle_GreenPotion", "Item_Bottle_RedPotion", "Item_Bottle", "Item_Bottle_Fairy", "Item_Bottle_Bug", "Item_Bottle_BluePotion",
    "Item_Bottle_Milk", "Item_Bottle_HalfMilk", "Item_Bottle_Fish", "Item_Bottle_BlueFire", "Item_Bottle_Poe", "Item_Bottle_BigPoe",
    "Item_Bottle_Letter", "Item_Bottle*", NULL,
    "Armor_Child_Mask_*", "Item_Mask_*", NULL,
};
static int item_rank(const char *name, int *group) {
    int g = 0;
    for (int i = 0; i < (int)(sizeof item_order / sizeof item_order[0]); i++) {
        const char *o = item_order[i];
        if (!o) { g++; continue; }
        const size_t n = strlen(o);
        if (o[n - 1] == '*' ? !strncmp(name, o, n - 1) : !strcmp(name, o)) return *group = g, i;
    }
    return *group = g, 1000;
}
static int item_before(int a, int b) {   /* item a before item b */
    int ga, gb;
    const int ra = item_rank(item_names[a], &ga), rb = item_rank(item_names[b], &gb);
    if (ga != gb) return ga < gb;
    if (ra != rb) return ra < rb;
    return strcmp(item_names[a], item_names[b]) < 0;
}
/* the slot key an item takes: one for every bottle (its content changes), the Hookshot's for the Longshot that replaces it */
static void key_of(const char *name, char *out, size_t cap) {
    if (!strncmp(name, "Item_Bottle", 11)) snprintf(out, cap, "Item_Bottle*");
    else if (!strcmp(name, "Item_Longshot")) snprintf(out, cap, "Item_Hookshot");
    else if (!strcmp(name, "Armor_GlassTruth")) snprintf(out, cap, "Item_GlassTruth");
    /* (the bow takes other names with an arrow on it: "Item_Arrow_Normal" for the bow with its plain arrows,
       "Pouch_Range_BowAndArrow_Ice" for the Ice Arrow while it is the one on the bow; seen at the Gerudo Fortress) */
    else if (!strcmp(name, "Item_Arrow_Normal") || !strcmp(name, "Pouch_Range_BowAndArrow_Normal")) snprintf(out, cap, "Weapon_Range_Bow");
    else if (!strncmp(name, "Pouch_Range_BowAndArrow_", 24)) snprintf(out, cap, "Item_Arrow_%s", name + 24);
    else snprintf(out, cap, "%s", name);
}

/* the pages a new player starts with: what the wheel showed before pages could be moved, each age's own */
static void layout_defaults(void) {
    /* (the player's own order, from the game's screens: slot 0 the top, then clockwise. The bottles fill their slots from
       the left over the top: wheel_layout) */
    static const char *const child[][MKWVK_WHEEL_SLOTS] = {
        {"Item_Boomerang", "", "Item_Bomb", "Item_Bombchu", "Weapon_Range_SlingShot", "Item_DekuStick", "Item_DekuNut", ""},
        {"Item_DinFire", "Item_NayruLove", "", "", "Item_MagicBean", "", "", "Item_FaroreWind"},
        {"Item_Bottle*", "Item_Bottle*", "Item_Bottle*", "", "Item_ScrubBusinessSeed", "", "Item_Bottle*", "Item_Bottle*"},
    };
    static const char *const adult[][MKWVK_WHEEL_SLOTS] = {
        {"Item_Hookshot", "Item_Arrow_Fire", "Item_Bomb", "Item_Arrow_Ice", "Item_Hammer", "Item_Arrow_Light", "Item_DekuNut", "Weapon_Range_Bow"},
        {"Item_DinFire", "Item_NayruLove", "", "", "Item_Bombchu", "", "", "Item_FaroreWind"},
        {"Item_Bottle*", "Item_Bottle*", "Item_Bottle*", "", "Item_ScrubBusinessSeed", "", "Item_Bottle*", "Item_Bottle*"},
    };
    memset(layouts, 0, sizeof layouts);
    for (int a = 0; a < 2; a++) {
        const char *const (*src)[MKWVK_WHEEL_SLOTS] = a ? adult : child;
        layouts[a].pages = 3;
        for (int p = 0; p < 3; p++)
            for (int s = 0; s < MKWVK_WHEEL_SLOTS; s++) snprintf(layouts[a].slot[p][s], sizeof layouts[a].slot[p][s], "%s", src[p][s]);
    }
}
static int layout_path(char *out, size_t cap) {   /* wheel_layout.txt in the game folder */
    char path[MAX_PATH];
    const DWORD len = hos_game_path_a(path, MAX_PATH);
    char *slash = len ? (strrchr(path, 92) ? strrchr(path, 92) : strrchr(path, 47)) : NULL;
    if (!slash) return 0;
    slash[1] = 0;
    snprintf(out, cap, "%swheel_layout.txt", path);
    return 1;
}
static void layout_save(void) {
    char path[MAX_PATH + 32];
    if (!layout_path(path, sizeof path)) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        hos_log("wheel: %s could not be written: the pages stay as they are until the game ends", path);
        return;
    }
    fprintf(f, "# Hookshot: the item wheel's pages (R held), one line a page: child or adult, the page's number, then its 8 slots\n"
               "# from the top, clockwise (- an empty slot, Item_Bottle* any bottle). In the wheel: Y on an item, then Y on the\n"
               "# slot to put it in. This file is written again at each move; delete it for the pages the game starts with.\n");
    for (int a = 0; a < 2; a++)
        for (int p = 0; p < layouts[a].pages; p++) {
            fprintf(f, "%s %d =", a ? "adult" : "child", p + 1);
            for (int s = 0; s < MKWVK_WHEEL_SLOTS; s++) fprintf(f, " %s", layouts[a].slot[p][s][0] ? layouts[a].slot[p][s] : "-");
            fputc('\n', f);
        }
    fclose(f);
}
static void layout_load(void) {
    layout_defaults();
    char path[MAX_PATH + 32];
    FILE *f = layout_path(path, sizeof path) ? fopen(path, "r") : NULL;
    if (!f) return;
    AgeLayout read[2];
    memset(read, 0, sizeof read);
    char line[1024];
    int lines = 0;
    while (fgets(line, sizeof line, f)) {
        char age[16];
        int page, at = 0;
        if (line[0] == '#' || sscanf(line, "%15s %d =%n", age, &page, &at) != 2 || !at) continue;
        const int a = !strcmp(age, "adult") ? 1 : !strcmp(age, "child") ? 0 : -1;
        if (a < 0 || page < 1 || page > LAYOUT_PAGES) continue;
        char *s = line + at;
        for (int k = 0; k < MKWVK_WHEEL_SLOTS; k++) {
            char word[64];
            int used = 0;
            if (sscanf(s, "%63s%n", word, &used) != 1) break;
            s += used;
            snprintf(read[a].slot[page - 1][k], sizeof read[a].slot[page - 1][k], "%s", strcmp(word, "-") ? word : "");
        }
        if (page > read[a].pages) read[a].pages = page;
        lines++;
    }
    fclose(f);
    for (int a = 0; a < 2; a++)
        if (read[a].pages) layouts[a] = read[a];   /* (an age the file leaves out keeps its defaults) */
    hos_log("wheel: %d pages from %s", lines, path);
}
/* the layout's pages with no slot named go: the pages after them move up */
static void layout_compact(AgeLayout *l) {
    int out = 0;
    for (int p = 0; p < l->pages; p++) {
        int any = 0;
        for (int s = 0; s < MKWVK_WHEEL_SLOTS; s++) any |= l->slot[p][s][0] != 0;
        if (!any) continue;
        if (out != p) memcpy(l->slot[out], l->slot[p], sizeof l->slot[p]);
        out++;
    }
    for (int p = out; p < l->pages; p++) memset(l->slot[p], 0, sizeof l->slot[p]);
    l->pages = out;
}

/* The pages as shown: the layout's (all of them), then the others', then, while an item is being moved, a new one.
   at[page * 8 + slot]: the item there, -1 none; info[page]: what the page is. Returns the pages */
static int wheel_layout(const AgeLayout *l, int moving, int at[LAYOUT_MAX], PageInfo info[PAGES_MAX]) {
    for (int i = 0; i < LAYOUT_MAX; i++) at[i] = -1;
    int sorted[MKWVK_WHEEL_MAX], used[MKWVK_WHEEL_MAX] = {0};
    for (int i = 0; i < item_n; i++) {
        int k = i;
        while (k > 0 && item_before(i, sorted[k - 1])) sorted[k] = sorted[k - 1], k--;
        sorted[k] = i;
    }
    int pages = 0;
    for (int p = 0; p < l->pages && pages < PAGES_MAX; p++, pages++) {
        info[pages].kind = PAGE_LAYOUT, info[pages].page = p;
        for (int k8 = 0; k8 < MKWVK_WHEEL_SLOTS; k8++) {
            const int s = (k8 + 6) % MKWVK_WHEEL_SLOTS;   /* (from the left slot over the top: the bottles in reading order) */
            if (!l->slot[p][s][0]) continue;
            for (int k = 0; k < item_n; k++) {   /* (the first item, in item_order's order, not placed yet) */
                char key[48];
                key_of(item_names[sorted[k]], key, sizeof key);
                if (used[sorted[k]] || strcmp(key, l->slot[p][s])) continue;
                used[sorted[k]] = 1, at[pages * MKWVK_WHEEL_SLOTS + s] = sorted[k];
                break;
            }
        }
    }
    int slot = MKWVK_WHEEL_SLOTS;
    for (int k = 0; k < item_n; k++) {
        if (used[sorted[k]]) continue;
        if (slot == MKWVK_WHEEL_SLOTS) {
            if (pages == PAGES_MAX) break;
            info[pages].kind = PAGE_OTHERS, info[pages].page = -1, pages++, slot = 0;
        }
        at[(pages - 1) * MKWVK_WHEEL_SLOTS + slot++] = sorted[k];
    }
    if (moving && l->pages < LAYOUT_PAGES && pages < PAGES_MAX) info[pages].kind = PAGE_NEW, info[pages].page = l->pages, pages++;
    return pages;
}
/* a move: the item at from (a page * 8 + slot as shown) put in to's slot. Into a slot of the layout's or a new page: the
   two slots swap (an item from the others' pages leaves to's key to the others'); onto the others' pages: it leaves its
   slot. Then the pages are written to wheel_layout.txt */
static void layout_move(AgeLayout *l, const int at[LAYOUT_MAX], const PageInfo info[PAGES_MAX], int pages, int from, int to) {
    const int fp = from / MKWVK_WHEEL_SLOTS, fs = from % MKWVK_WHEEL_SLOTS, tp = to / MKWVK_WHEEL_SLOTS, ts = to % MKWVK_WHEEL_SLOTS;
    if (from < 0 || to < 0 || fp >= pages || tp >= pages || at[from] < 0 || from == to) return;
    char key[48];
    key_of(item_names[at[from]], key, sizeof key);
    if (info[tp].kind == PAGE_OTHERS) {
        if (info[fp].kind != PAGE_LAYOUT) return;
        l->slot[info[fp].page][fs][0] = 0;
    } else {
        const int lp = info[tp].page;
        if (info[tp].kind == PAGE_NEW) {
            if (l->pages >= LAYOUT_PAGES) return;
            memset(l->slot[l->pages], 0, sizeof l->slot[l->pages]);
            l->pages++;
        }
        char old[40];
        memcpy(old, l->slot[lp][ts], sizeof old);
        if (info[fp].kind == PAGE_LAYOUT) memcpy(l->slot[info[fp].page][fs], old, sizeof old);
        snprintf(l->slot[lp][ts], sizeof l->slot[lp][ts], "%s", key);
    }
    layout_compact(l);
    if (wheel_trace()) hos_log("wheel: %s moved from slot %d of page %d to slot %d of page %d", item_names[at[from]], fs, fp + 1, ts, tp + 1);
    layout_save();
}

/* The survey: the row loads only about 4 items behind its cursor to 5 ahead, so places it has not shown are not known.
   Hidden behind the wheel's backdrop, the row is turned (the same taps) to the nearest stop that loads an unknown place:
   while the wheel does not know as many items as the row has (the first opening, an item gained or used up), and to
   find a chosen item whose place is not known. The stop for the nearest unknown place (a margin of one inside the
   window), the cursor itself while one is in the window but not loaded yet, -1 when all are known */
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

static void wheel_tick(Ctx *c, uint64_t screen, int opened) {
    static int last_cursor = -1, seq_seen = -1, open_cursor = -1;   /* (open_cursor: the cursor as the opening settled) */
    static int64_t opened_at;
    static char home_name[48], choice_name[48], shown_name[48];   /* (shown: the item chosen last this opening, kept
                                                                     on the wheel after it is equipped, until it closes) */
    const int64_t now = hos_now_ns();
    const uint64_t node = peek64(screen + 0x40);
    int32_t cursor = -1;
    if (node) peek(node + 0x44, &cursor, 4);
    const int length = node ? row_length(node) : -1;
    if (opened) {
        /* (the item the row was left on at the last close is the one equipped: the game put it first) */
        if (last_cursor > 0 && last_cursor != open_cursor && length == row_names_count) {   /* (left where it opened: nothing equipped) */
            if (wheel_trace()) hos_log("wheel: %s was equipped from place %d: now first", row_names[last_cursor], last_cursor);
            row_to_front(last_cursor);
        }
        last_cursor = -1, open_cursor = -1, home_name[0] = 0, shown_name[0] = 0, opened_at = now;
        __atomic_add_fetch(&wheel_opening, 1, __ATOMIC_ACQ_REL);
    }
    const int count = length < 0 ? 0 : length < MKWVK_WHEEL_MAX ? length : MKWVK_WHEEL_MAX;
    /* (not in the opening's first 150 ms: the row's panes still show it as it was before the game put the item equipped
       last first, and learning them would undo row_to_front) */
    if (length >= 0 && now - opened_at >= 150000000LL) row_learn(node, length);
    if (cursor >= 0) last_cursor = cursor;
    if (open_cursor < 0 && cursor >= 0 && now - opened_at >= 150000000LL) open_cursor = cursor;
    uint32_t known = 0;
    for (int i = 0; i < count && i < 32; i++)
        if (row_names[i][0]) known |= 1u << i;
    /* (the survey's places: only while the wheel lacks some of the row's items; a chosen item's place unknown is searched
       for by the menu tick below) */
    __atomic_store_n(&row_known, item_n == count ? 0xffffffffu : known, __ATOMIC_RELEASE);
    __atomic_store_n(&row_count, count, __ATOMIC_RELEASE);
    __atomic_store_n(&row_cursor, cursor, __ATOMIC_RELEASE);
    /* the item equipped: the game's own answer (the item on ZR), asked as the opening starts; else the cursor's as the
       opening settles (the survey turns the row after that) */
    static int home_asked;
    if (opened) home_asked = cy_zr_name(c, home_name, sizeof home_name);
    if (!home_asked && now - opened_at < 200000000LL && cursor >= 0 && cursor < MKWVK_WHEEL_MAX && row_names[cursor][0])
        memcpy(home_name, row_names[cursor], sizeof home_name);

    /* the pages: the layout of Link's age (rx_option(OOT_CHEAT_AGE_ID): 0 the child, 1 the adult, -1 no save: the adult's) */
    static int layouts_read, drop_seen;
    if (!layouts_read) layouts_read = 1, layout_load();
    int rx_option(int id, int otherwise);
    AgeLayout *lay = &layouts[rx_option(OOT_CHEAT_AGE_ID, 1) == 0 ? 0 : 1];
    int at[LAYOUT_MAX];
    PageInfo info[PAGES_MAX];
    const int drop = __atomic_load_n(&drop_seq, __ATOMIC_ACQUIRE);
    if (drop != drop_seen) {   /* (an item put down: moved in the pages it was moved on, the ones shown while one is held) */
        drop_seen = drop;
        const int req = __atomic_load_n(&drop_req, __ATOMIC_ACQUIRE);
        const int shown = wheel_layout(lay, 1, at, info);
        layout_move(lay, at, info, shown, req >> 16, req & 0xffff);
    }
    const int held = __atomic_load_n(&move_from, __ATOMIC_ACQUIRE);
    const int pages = wheel_layout(lay, held >= 0, at, info);
    int page = __atomic_load_n(&wheel_page, __ATOMIC_ACQUIRE);
    if (page >= pages) page = pages - 1;
    if (page < 0) page = 0;
    __atomic_store_n(&page_count, pages, __ATOMIC_RELEASE);

    /* the item chosen: its name taken once a choice (the layout may change while the survey finds more), its place
       looked up each update */
    const int seq = __atomic_load_n(&choice_seq, __ATOMIC_ACQUIRE);
    if (seq != seq_seen) {
        const int key = __atomic_load_n(&choice_key, __ATOMIC_ACQUIRE);
        choice_name[0] = 0;
        if (key >= 0 && key < LAYOUT_MAX && at[key] >= 0) memcpy(choice_name, item_names[at[key]], sizeof choice_name);
        if (choice_name[0]) memcpy(shown_name, choice_name, sizeof shown_name);
        /* (its number in the game: the wheel puts it on ZR itself once the row has closed, the row left where it was) */
        const int item = choice_name[0] ? cy_item_of_name(c, choice_name) : -1;
        if (item >= 0) memcpy(equip_name, choice_name, sizeof equip_name);
        __atomic_store_n(&choice_item, item, __ATOMIC_RELEASE);
        if (wheel_trace() && choice_name[0]) hos_log("wheel: %s is item %d in the game%s", choice_name, item, item < 0 ? ": the row is turned to it" : "");
        seq_seen = seq;
    }
    /* (a choice of an empty slot, the layout changed meanwhile: nothing to turn to) */
    int row = -1, here = 0, lost = !choice_name[0] && __atomic_load_n(&choice_key, __ATOMIC_ACQUIRE) >= 0;
    if (choice_name[0]) {
        row = row_find(choice_name);
        here = row >= 0 && row == cursor;
        if (row < 0) {   /* (not seen at its place yet: on through the places not known) */
            row = survey_stop(cursor, count, known);
            lost = row < 0;
        }
    }
    __atomic_store_n(&choice_row, row, __ATOMIC_RELEASE);
    __atomic_store_n(&choice_here, here, __ATOMIC_RELEASE);
    __atomic_store_n(&choice_lost, lost, __ATOMIC_RELEASE);
    __atomic_store_n(&choice_ack, seq, __ATOMIC_RELEASE);

    MkwVkWheel next;
    memset(&next, 0, sizeof next);
    next.open = 1;
    next.count = item_n ? MKWVK_WHEEL_SLOTS : 0;
    next.page = page, next.pages = pages;
    next.cursor = next.pick = next.held = -1;
    for (int p = 0; p < pages && p < (int)sizeof next.page_kind; p++) next.page_kind[p] = (uint8_t)info[p].kind;
    if (held >= 0 && held < LAYOUT_MAX && at[held] >= 0) {
        snprintf(next.held_name, sizeof next.held_name, "%s", item_names[at[held]]);
        if (held / MKWVK_WHEEL_SLOTS == page) next.held = held % MKWVK_WHEEL_SLOTS;
    }
    uint32_t mask = 0;
    for (int s = 0; s < MKWVK_WHEEL_MAX; s++) next.amount[s] = -1;
    for (int s = 0; s < MKWVK_WHEEL_SLOTS; s++) {
        const int key = page * MKWVK_WHEEL_SLOTS + s, i = key < LAYOUT_MAX ? at[key] : -1;
        if (i < 0) continue;
        mask |= 1u << s;
        snprintf(next.name[s], sizeof next.name[s], "%s", item_names[i]);
        label_of(item_names[i], next.label[s], sizeof next.label[s]);
        next.amount[s] = amount_of(item_names[i]);
        if (home_name[0] && !strcmp(item_names[i], home_name)) next.cursor = s;
        if (shown_name[0] && !strcmp(item_names[i], shown_name)) next.pick = s;
    }
    __atomic_store_n(&page_mask, (uint32_t)page << 8 | mask, __ATOMIC_RELEASE);
    const int pick = __atomic_load_n(&wheel_pick, __ATOMIC_ACQUIRE);
    if (pick >= 0 && pick < MKWVK_WHEEL_SLOTS) next.pick = pick;   /* (an empty slot too: where a held item would go) */
    else if (held >= 0) next.pick = -1;
    static int ticks_open;   /* (the items once an opening, 10 updates in: the row fills its panes over its first frames) */
    if (opened) ticks_open = 0;
    if (++ticks_open == 10) {
        char list[MKWVK_WHEEL_MAX * 34] = "";
        size_t n = 0;
        for (int i = 0; i < count && n < sizeof list - 40; i++) {
            char label[32] = "?";
            if (row_names[i][0]) label_of(row_names[i], label, sizeof label);
            n += (size_t)snprintf(list + n, sizeof list - n, "%s%s%s", i ? ", " : "", i == cursor ? "*" : "", label);
        }
        hos_log("wheel: the row (screen %#llx) shows %d items (the wheel knows %d, %d pages): %s", (unsigned long long)screen,
                count, item_n, pages, list);
    }
    publish(&next);
}

/* ---- Cycle Items --------------------------------------------------------------------------------------------------
   The Gameplay tab's Cycle items row (HOS_OPT_CYCLE_ITEMS; HOS_CYCLE_ITEMS=0..3 over it): in play, a tap of R puts the
   next item of the ZR item's group on ZR, a tap of L the one before, without the wheel; held past HOS_WHEEL_HOLD_MS (250
   by default) R or L opens the wheel as before. The groups: the bow and its three arrows, the bottles (the pouch's five
   bottle slots, in their order), Din's Fire, Farore's Wind and Nayru's Love; with 2 also the Bomb and the Bombchu, with 3
   also the Hookshot (or the Longshot) and the Megaton Hammer, and the child's Slingshot and Boomerang. Only items the game would list in its own row now count
   (owned, usable at Link's age).
   The game's calls (the lifted addresses, and the item numbers, PouchMgr's AssignableItem: 0 DekuStick .. 53 NormalLure)
   are those PR #2151 (item assignable buttons) found: PouchMgr::getAssignableItem and setAssignableItem, and the walk of
   the pouch the quick-switch makes its list with. The change is made on the game's thread, on the HUD screen's update */
#define CY_POUCH_SLOT 0x84e98e28ULL      /* PouchMgr::sInstance */
#define CY_UI_SLOT 0x84e9cfa8ULL         /* game::ui::UIModule: +0x5c0 bit 0 set for the child */
#define CY_HUD_VTABLE 0x84c0b710ULL      /* UIPlayerStatusScreen: play's HUD */
#define CY_PLUS_VTABLE 0x84c0baa8ULL     /* UIPlusMenuScreen: the pause menu */
#define CY_OVERLAY_VTABLE 0x84c08aa0ULL  /* UIMenuOverlayScreen: over it */
#define CY_GETTER 0x8019b6e0ULL          /* PouchMgr::getAssignableItem(mgr, w1 age, w2 isDirect) */
#define CY_SETTER 0x81231c00ULL          /* PouchMgr::setAssignableItem(mgr, w1 age, w2 item, w3 reason: 1 the quick-switch's) */
#define CY_POUCH_ARRAY 0x80a8f690ULL     /* PouchMgr::getPouchArray(mgr, w1 kind): 0 the ItemArray */
#define CY_ARRAY_COUNT 0x80021a20ULL     /* an array's count */
#define CY_SLOT_ACTOR 0x80a8ef20ULL      /* PouchMgr::getCurrentActorOfSlot(mgr, w1 kind, w2 slot, x3 s32 *bottle, x4 u32 *entry) */
#define CY_ACTOR_SHORT 0x8002a4c0ULL     /* PouchItemParam::getActorShort(param): a char ** */
#define CY_ITEM_OF_ACTOR 0x833c6860ULL   /* PouchMgr::getAssignableItemOfActor(mgr, x1 char **): -1 none */
#define CY_LISTABLE 0x805b6af0ULL        /* PouchMgr::canListInShortCut(mgr, x1 char **, w2 bottle, x3 param, w4 age) */
#define CY_ITEMS 54

static volatile int64_t hud_tick_ns, pause_tick_ns;   /* (the HUD's and the pause menu's last updates: play or not) */
static volatile int cycle_seq, cycle_dir;             /* (the input side's taps: +1 R, -1 L) */

static int cycle_level(void) {   /* 0 off, 1 the arrows, bottles and spells, 2 and bombs, 3 and the hookshot and hammer */
    int rx_option(int id, int otherwise);
    static int forced = -2;
    if (forced == -2) {
        const char *e = getenv("HOS_CYCLE_ITEMS");
        forced = e ? atoi(e) : -1;
    }
    const int v = forced >= 0 ? forced : rx_option(HOS_OPT_CYCLE_ITEMS, 1);
    return v < 0 ? 0 : v > 3 ? 3 : v;
}
/* play now: the HUD updated within 100 ms, the pause menu not within 150 */
static int in_play(int64_t now) {
    const int64_t hud = __atomic_load_n(&hud_tick_ns, __ATOMIC_ACQUIRE), pause = __atomic_load_n(&pause_tick_ns, __ATOMIC_ACQUIRE);
    return hud && now - hud < 100000000LL && (!pause || now - pause > 150000000LL);
}

/* a call of a game function on the game's thread: x[0..n) in, x0 out; the Ctx put back as it was (PR #2151's ib_guest) */
static uint64_t cy_guest(Ctx *c, uint64_t fn, int n, const uint64_t *x) {
    A64Fn f = a64_lookup(fn);
    if (!f || n < 0 || n > 8) return 0;
    const Ctx saved = *c;
    for (int i = 0; i < n; i++) c->x[i] = x[i];
    c->x[30] = 0;
    c->x[31] = (saved.x[31] - 0x200) & ~0xfULL;   /* (its frames below the host's records) */
    f(c);
    const uint64_t r = c->x[0];
    *c = saved;
    return r;
}
static int cy_bottle(int item) { return (item >= 14 && item <= 17) || item == 50; }
/* a read anywhere in mapped guest memory (the pouch's arrays and the stack are outside valid()'s image and heap) */
static uint64_t cy_peek(uint64_t a, unsigned n) {
    uint64_t v = 0;
    if (a >= 0x100000 && n <= 8 && hos_mem_region(a, n)) memcpy(&v, GP(a), n);
    return v;
}
/* the items the game would list in its row now for the age (bits by item); 0 when the pouch cannot be walked now */
static int cy_listable(Ctx *c, uint64_t mgr, int age, uint64_t *out) {
    static const uint64_t calls[] = {CY_POUCH_ARRAY, CY_ARRAY_COUNT, CY_SLOT_ACTOR, CY_ACTOR_SHORT, CY_ITEM_OF_ACTOR, CY_LISTABLE};
    for (size_t i = 0; i < sizeof calls / sizeof calls[0]; i++)
        if (!a64_lookup(calls[i])) return 0;
    const uint64_t scr = (c->x[31] - 0x100) & ~0xfULL;   /* (the host's records: below the sp, above the calls' frames) */
    const uint64_t kind[2] = {mgr, 0};
    const uint64_t array = cy_guest(c, CY_POUCH_ARRAY, 2, kind);
    const int n = array ? (int32_t)cy_guest(c, CY_ARRAY_COUNT, 1, &array) : 0;
    if (n < 1 || n > 64) return 0;
    uint64_t v = 0;
    for (int i = 0; i < n; i++) {
        recomp_store32_sp(c, scr, 0xffffffffu), recomp_store32_sp(c, scr + 4, 0);
        const uint64_t at[5] = {mgr, 0, (uint32_t)i, scr, scr + 4};
        const uint64_t actor = cy_guest(c, CY_SLOT_ACTOR, 5, at);
        const uint64_t name = actor ? cy_guest(c, CY_ACTOR_SHORT, 1, &actor) : 0;
        if (!name || !cy_peek(cy_peek(name, 8), 1)) continue;   /* (an empty slot) */
        const int32_t bottle = (int32_t)cy_peek(scr, 4);
        const uint64_t of[2] = {mgr, name};
        const int item = cy_bottle(bottle) ? bottle : (int32_t)cy_guest(c, CY_ITEM_OF_ACTOR, 2, of);
        if (item < 0 || item >= CY_ITEMS) continue;
        const uint64_t list[5] = {mgr, name, (uint32_t)bottle, actor, (uint64_t)age};
        if (cy_guest(c, CY_LISTABLE, 5, list) & 1) v |= 1ULL << item;
    }
    *out = v;
    return 1;
}
/* the item the pouch holds under a name (the wheel's, canonical()'s): its number, a bottle's its slot's; -1 none */
static int cy_item_of_name(Ctx *c, const char *want) {
    static const uint64_t calls[] = {CY_POUCH_ARRAY, CY_ARRAY_COUNT, CY_SLOT_ACTOR, CY_ACTOR_SHORT, CY_ITEM_OF_ACTOR};
    for (size_t i = 0; i < sizeof calls / sizeof calls[0]; i++)
        if (!a64_lookup(calls[i])) return -1;
    const uint64_t mgr = cy_peek(CY_POUCH_SLOT, 8);
    if (!mgr || !want || !want[0]) return -1;
    const uint64_t scr = (c->x[31] - 0x100) & ~0xfULL;
    const uint64_t kind[2] = {mgr, 0};
    const uint64_t array = cy_guest(c, CY_POUCH_ARRAY, 2, kind);
    const int n = array ? (int32_t)cy_guest(c, CY_ARRAY_COUNT, 1, &array) : 0;
    for (int i = 0; i < n && i < 64; i++) {
        recomp_store32_sp(c, scr, 0xffffffffu), recomp_store32_sp(c, scr + 4, 0);
        const uint64_t at[5] = {mgr, 0, (uint32_t)i, scr, scr + 4};
        const uint64_t actor = cy_guest(c, CY_SLOT_ACTOR, 5, at);
        const uint64_t name = actor ? cy_guest(c, CY_ACTOR_SHORT, 1, &actor) : 0;
        const uint64_t text = name ? cy_peek(name, 8) : 0;
        if (!text) continue;
        char got[64];
        size_t k = 0;
        for (; k + 1 < sizeof got && hos_mem_region(text + k, 1); k++)
            if (!(got[k] = *(const char *)GP(text + k))) break;
        got[k] = 0;
        canonical(got, sizeof got);
        if (strcmp(got, want)) continue;
        const int32_t bottle = (int32_t)cy_peek(scr, 4);
        const uint64_t of[2] = {mgr, name};
        const int item = cy_bottle(bottle) ? bottle : (int32_t)cy_guest(c, CY_ITEM_OF_ACTOR, 2, of);
        return item >= 0 && item < CY_ITEMS ? item : -1;
    }
    return -1;
}
/* the name (canonical()'s) of the item on ZR now, from the pouch slot that holds it; 0 when not found */
static int cy_zr_name(Ctx *c, char *out, size_t cap) {
    static const uint64_t calls[] = {CY_GETTER, CY_POUCH_ARRAY, CY_ARRAY_COUNT, CY_SLOT_ACTOR, CY_ACTOR_SHORT, CY_ITEM_OF_ACTOR};
    for (size_t i = 0; i < sizeof calls / sizeof calls[0]; i++)
        if (!a64_lookup(calls[i])) return 0;
    const uint64_t mgr = cy_peek(CY_POUCH_SLOT, 8), ui = cy_peek(CY_UI_SLOT, 8);
    if (!mgr || !ui) return 0;
    const uint64_t g[3] = {mgr, cy_peek(ui + 0x5c0, 1) & 1, 1};
    const int cur = (int32_t)cy_guest(c, CY_GETTER, 3, g);
    if (cur < 0 || cur >= CY_ITEMS) return 0;
    const uint64_t scr = (c->x[31] - 0x100) & ~0xfULL;
    const uint64_t kind[2] = {mgr, 0};
    const uint64_t array = cy_guest(c, CY_POUCH_ARRAY, 2, kind);
    const int n = array ? (int32_t)cy_guest(c, CY_ARRAY_COUNT, 1, &array) : 0;
    for (int i = 0; i < n && i < 64; i++) {
        recomp_store32_sp(c, scr, 0xffffffffu), recomp_store32_sp(c, scr + 4, 0);
        const uint64_t at[5] = {mgr, 0, (uint32_t)i, scr, scr + 4};
        const uint64_t actor = cy_guest(c, CY_SLOT_ACTOR, 5, at);
        const uint64_t name = actor ? cy_guest(c, CY_ACTOR_SHORT, 1, &actor) : 0;
        const uint64_t text = name ? cy_peek(name, 8) : 0;
        if (!text) continue;
        const int32_t bottle = (int32_t)cy_peek(scr, 4);
        const uint64_t of[2] = {mgr, name};
        const int item = cy_bottle(bottle) ? bottle : (int32_t)cy_guest(c, CY_ITEM_OF_ACTOR, 2, of);
        if (item != cur) continue;
        size_t k = 0;
        for (; k + 1 < cap && hos_mem_region(text + k, 1); k++)
            if (!(out[k] = *(const char *)GP(text + k))) break;
        out[k] = 0;
        canonical(out, cap);
        return out[0] != 0;
    }
    return 0;
}
/* the row's places after the game put an item first (setAssignableItem with the quick-switch's reason sorts its row as a
   choice there does): the item's place moved to the front, a bottle's (whose content is not known here) all forgotten */
static void cycle_to_front(int item) {
    static const struct { int item; const char *name; } names[] = {
        {30, "Weapon_Range_Bow"}, {46, "Item_Arrow_Fire"}, {47, "Item_Arrow_Ice"}, {48, "Item_Arrow_Light"},
        {10, "Item_DinFire"}, {11, "Item_FaroreWind"}, {12, "Item_NayruLove"}, {2, "Item_Bomb"}, {9, "Item_Bombchu"},
        {5, "Item_Hookshot"}, {6, "Item_Longshot"}, {7, "Item_Hammer"}, {4, "Weapon_Range_SlingShot"}, {3, "Item_Boomerang"},
    };
    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++)
        if (names[k].item == item) {
            const int at = row_find(names[k].name);
            if (at > 0) row_to_front(at);
            return;
        }
    if (row_names_count > 0) memset(row_names, 0, sizeof row_names);   /* (a bottle: its place is looked for when chosen) */
}
/* the wheel's choice put on ZR (the game's row closed, left where it opened): the game puts it first in the row, as a choice
   there does, and the row's places follow */
static void equip_now(Ctx *c, int item) {
    const uint64_t mgr = cy_peek(CY_POUCH_SLOT, 8), ui = cy_peek(CY_UI_SLOT, 8);
    if (!mgr || !ui || item < 0 || item >= CY_ITEMS || !a64_lookup(CY_SETTER)) return;
    const uint64_t s[4] = {mgr, cy_peek(ui + 0x5c0, 1) & 1, (uint64_t)item, 1};
    cy_guest(c, CY_SETTER, 4, s);
    const int at = row_find(equip_name);
    if (at > 0) row_to_front(at);
    if (wheel_trace()) hos_log("wheel: %s (item %d) put on ZR", equip_name, item);
}
/* a tap: the ZR item's group's next (dir +1) or previous (-1) listable item put on ZR */
static void cycle_now(Ctx *c, int dir, int level) {
    static const struct { int level; int items[6]; } groups[] = {
        {1, {30, 46, 47, 48, -1}},       /* the bow, Fire, Ice and Light Arrows */
        {1, {14, 15, 16, 17, 50, -1}},   /* the bottles */
        {1, {10, 11, 12, -1}},           /* Din's Fire, Farore's Wind, Nayru's Love */
        {2, {2, 9, -1}},                 /* the Bomb, the Bombchu */
        {3, {5, 6, 7, -1}},              /* the Hookshot or the Longshot, the Megaton Hammer */
        {3, {4, 3, -1}},                 /* the child's Slingshot and Boomerang */
    };
    const uint64_t mgr = cy_peek(CY_POUCH_SLOT, 8), ui = cy_peek(CY_UI_SLOT, 8);
    if (!mgr || !ui || !a64_lookup(CY_GETTER) || !a64_lookup(CY_SETTER)) return;
    const int age = (int)(cy_peek(ui + 0x5c0, 1) & 1);   /* (PouchMgr's: 0 the adult, 1 the child) */
    const uint64_t g[3] = {mgr, (uint64_t)age, 1};
    const int cur = (int32_t)cy_guest(c, CY_GETTER, 3, g);
    uint64_t ok = 0;
    if (!cy_listable(c, mgr, age, &ok)) {
        if (wheel_trace()) hos_log("wheel: cycle: the pouch could not be walked now");
        return;
    }
    if (wheel_trace()) hos_log("wheel: cycle: age %d, item on ZR %d, items listable %#llx", age, cur, (unsigned long long)ok);
    for (size_t k = 0; k < sizeof groups / sizeof groups[0]; k++) {
        if (groups[k].level > level) continue;
        int n = 0, at = -1;
        while (groups[k].items[n] >= 0) {
            if (groups[k].items[n] == cur) at = n;
            n++;
        }
        if (at < 0) continue;
        for (int step = 1; step < n; step++) {
            const int item = groups[k].items[((at + dir * step) % n + n) % n];
            if (!(ok >> item & 1)) continue;
            const uint64_t s[4] = {mgr, (uint64_t)age, (uint64_t)item, 1};
            cy_guest(c, CY_SETTER, 4, s);
            cycle_to_front(item);
            void hos_audio_cue(const char *name);
            hos_audio_cue("SE_Sys_ShortCut_CursolMove_00");
            if (wheel_trace()) hos_log("wheel: cycle %s: item %d -> %d", dir > 0 ? "R" : "L", cur, item);
            return;
        }
        if (wheel_trace()) hos_log("wheel: cycle: item %d has nothing else of its group to go to", cur);
        return;
    }
    if (wheel_trace()) hos_log("wheel: cycle: item %d is in no group", cur);
}

/* the slot in direction (x, y) (y up): slot 0 at the top, the rest clockwise, each a 360/count wedge round its own
   direction */
static int slot_toward(double x, double y, int count) {
    double a = atan2(x, y);   /* (0 up, clockwise positive) */
    if (a < 0) a += 2 * WHEEL_PI;
    return (int)floor(a / (2 * WHEEL_PI / count) + 0.5) % count;
}
/* how far direction (x, y) is from slot s's own, in degrees */
static double slot_off(double x, double y, int s, int count) {
    double d = atan2(x, y) - 2 * WHEEL_PI * s / count;
    while (d > WHEEL_PI) d -= 2 * WHEEL_PI;
    while (d < -WHEEL_PI) d += 2 * WHEEL_PI;
    return fabs(d) * 180 / WHEEL_PI;
}

/* the slot under the mouse pointer (client pixels px, py in a w x h client), -1 in the wheel's hub. The wheel is centred
   on the window, its unit a hundredth of the height and its hub 12 units across (gpu/vk/vk_menu.cpp wheel_geometry) */
static int slot_under(int32_t px, int32_t py, int32_t w, int32_t h, int count) {
    if (count <= 0 || w <= 0 || h <= 0) return -1;
    const double u = h / 100 > 1 ? h / 100 : 1, dx = px - w / 2, dy = py - h / 2;
    if (dx * dx + dy * dy <= 12.0 * u * 12.0 * u) return -1;
    return slot_toward(dx, -dy, count);
}

/* The game's controller state, before the game sees it (nn_hid.c npad_states). While the row is shown:
   - the left stick is the wheel's: 8 slots 45 degrees apart, a slot taken once the stick is past 55% of its travel and
     kept until it points 30 degrees from that slot's middle (a stick between two slots does not flicker), the pick kept
     when the stick is let go;
   - the right stick points as the left one does (the one pushed further wins);
   - the D-pad's Left and Right, and ZL and ZR, turn the page (the game sees none of them);
   - a pick chooses its item; the row is turned to the item's place with D-pad taps, and R stays held for the game until
     the row is there (at most 4 s after the player let go, and only while the row is shown: the game closing it ends the
     R and the taps at once, row_gone).
   The mouse (host/display.c): while the wheel is open it is let go and its pointer shown. Moved, the item under it is
   the pick as the stick's is (until the stick is pushed again); a left click on an item picks it and closes the wheel
   (R let go for the game however it is held, and kept from it until the player lets go of it too). */
/* The sounds (nn_audio.c): the game's row plays its own open, select and close; its scroll tick, once a tap of the hidden
   row, is silent while the row updates (and 0.4 s after), and the wheel ticks once each new slot it points at instead. */
int hos_wheel_hush(void) {
    const int64_t last = __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE);
    return last && hos_now_ns() - last < 400000000LL;
}

static void choose(int key) {   /* (the input thread) a new choice for the menu tick, -1 none */
    __atomic_store_n(&choice_key, key, __ATOMIC_RELEASE);
    __atomic_add_fetch(&choice_seq, 1, __ATOMIC_ACQ_REL);
}

uint64_t hos_wheel_input(uint64_t buttons, int32_t stick[4]) {
    void hos_wheel_mouse_open(int open);   /* display.c */
    void hos_audio_cue(const char *name);  /* nn_audio.c */
    int hos_wheel_pointer(int32_t *x, int32_t *y, int32_t *w, int32_t *h, int *clicked, int32_t *click_x, int32_t *click_y);
    static int opening_seen = -1;
    static int chosen = -1;   /* the key chosen this opening, -1 none */
    static int64_t r_let_go_at;
    static int tap_dir, r_kept;
    static int was_open, click_close, r_block, mouse_live;
    static int32_t ref_x, ref_y;
    static int stick_slot = -1, cued = -1;
    static int moving = -1;   /* the item being moved (Y): page * 8 + slot as shown, -1 none (move_from, for the menu tick) */
    static int prev_moving_b; /* (B pressed to put an item back: kept from the game until let go) */
    static int pointed = -1;  /* the slot pointed at last on this page, kept when the stick is let go (Y acts on it) */
    static uint64_t prev_buttons;
    /* (home: the row's cursor as the opening settles, dropped for good at the first pick. It follows the cursor for the
       opening's first 200 ms, and the survey waits as long: after the row's items changed, its first update still gave
       the old cursor) */
    static int home = -1, home_taken;
    static int start_home = -1;   /* (the cursor the opening settled on: kept after a pick, where the row is left when the wheel equips) */
    static int64_t survey_until, opened_at;
    /* a tap of the hidden row: held for tap_frames of the game's controller reads, then as many let go
       (HOS_WHEEL_TAP_FRAMES, 1 to 8, 1 by default: 33 ms a tap at 60 fps, 17 places in 0.57 s,
       every tap taken). Counted in reads, not in time: taps of 35 ms
       were missed at 43 fps, where they were shorter than two frames. The game holding R or the D-pad itself repeats
       slower (330 ms, then 83 ms a place) */
    static int tap_frames, tap_down_n, tap_wait_n;
    static int64_t last_read;
    if (!tap_frames) {
        const char *e = getenv("HOS_WHEEL_TAP_FRAMES");
        const int n = e ? atoi(e) : 1;
        tap_frames = n < 1 ? 1 : n > 8 ? 8 : n;
    }
    const int64_t now = hos_now_ns();
    const int64_t last = __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE);
    /* (shown: updated within 100 ms; while R is held, the game's or ours, within 500 ms: a game stall is not a close) */
    const int open = last && !row_gone(last) && (now - last < ROW_OPEN_NS || (r_kept && now - last < ROW_STALL_NS));
    const int opening = __atomic_load_n(&wheel_opening, __ATOMIC_ACQUIRE);
    if (open != was_open) {
        if (!open && wheel_trace())
            hos_log("wheel: the row closes (%s; R %s)", row_gone(last) ? "the game went on without it" : "no update",
                    r_kept ? "kept held: let go now" : "the player's");
        if (!open) __atomic_store_n(&wheel_closed, 1, __ATOMIC_RELEASE);   /* (for the menu tick: a new opening next) */
        was_open = open, hos_wheel_mouse_open(open);
    }
    int32_t mx = 0, my = 0, mw = 0, mh = 0, cx = 0, cy = 0;
    int clicked = 0;
    const int mouse_ok = hos_wheel_pointer(&mx, &my, &mw, &mh, &clicked, &cx, &cy);
    if (!open) {
        /* Cycle Items: in play, R or L kept from the game while it may still be a tap; let go before HOS_WHEEL_HOLD_MS a
           tap (R the next item of the ZR item's group, L the one before), held longer it reaches the game (the row and
           the wheel open). A press that began out of play, or with the row off, goes to the game as it is */
        {
            static int64_t hold_ns, held_since;
            static uint64_t held_btn;
            static int passing;
            if (!hold_ns) {
                const char *e = getenv("HOS_WHEEL_HOLD_MS");
                const int ms = e ? atoi(e) : 250;
                hold_ns = (int64_t)(ms < 50 ? 50 : ms > 2000 ? 2000 : ms) * 1000000LL;
            }
            const uint64_t sh = buttons & HOLD_BUTTONS;
            if (!sh) {
                if (held_since && now - held_since < hold_ns) {
                    __atomic_store_n(&cycle_dir, held_btn & (1ULL << B_R) ? 1 : -1, __ATOMIC_RELEASE);
                    __atomic_add_fetch(&cycle_seq, 1, __ATOMIC_ACQ_REL);
                    if (wheel_trace()) hos_log("wheel: %s tapped (%.0f ms)", held_btn & (1ULL << B_R) ? "R" : "L", (now - held_since) / 1e6);
                }
                held_since = 0, passing = 0;
            } else if (!held_since && !passing) {   /* (a new press) */
                if (cycle_level() > 0 && in_play(now) && !r_block) held_since = now, held_btn = sh;
                else passing = 1;
            }
            if (held_since && sh) {
                if (now - held_since >= hold_ns) held_since = 0, passing = 1;   /* (a hold: the game has it from now) */
                else buttons &= ~HOLD_BUTTONS;
            }
        }
        if (chosen >= 0 || tap_dir) chosen = -1, tap_dir = 0, choose(-1);
        if (moving >= 0) moving = -1, __atomic_store_n(&move_from, -1, __ATOMIC_RELEASE);   /* (closed while held: not moved) */
        __atomic_store_n(&wheel_pick, -1, __ATOMIC_RELEASE);
        stick_slot = -1, cued = -1, pointed = -1;
        r_let_go_at = 0, r_kept = 0;
        if (home >= 0) home = -1, __atomic_store_n(&wheel_home, -1, __ATOMIC_RELEASE);
        /* (closed by a click: R kept from the game until the player lets go of it, or the row would open again) */
        if (click_close) r_block = 1, click_close = 0;
        if (r_block) {
            if (buttons & HOLD_BUTTONS) buttons &= ~HOLD_BUTTONS;
            else r_block = 0;
        }
        prev_buttons = buttons;
        return buttons;
    }
    if (opening != opening_seen) {
        opening_seen = opening, chosen = -1, tap_dir = 0, tap_down_n = tap_wait_n = 0, r_let_go_at = 0;
        click_close = 0, mouse_live = 0, ref_x = mx, ref_y = my;
        stick_slot = -1, cued = -1, pointed = -1, prev_buttons = buttons;   /* (a D-pad held from play is no turn) */
        home = -1, home_taken = 0, start_home = -1, survey_until = now + 3000000000LL, opened_at = now;
        __atomic_store_n(&wheel_home, -1, __ATOMIC_RELEASE);
        choose(-1);
        moving = -1, __atomic_store_n(&move_from, -1, __ATOMIC_RELEASE);
    }
    const int count = __atomic_load_n(&row_count, __ATOMIC_ACQUIRE);
    const int cursor = __atomic_load_n(&row_cursor, __ATOMIC_ACQUIRE);
    const int r_pressed = (buttons & HOLD_BUTTONS) != 0;   /* (the player's own R or L) */
    const int settled_in = now - opened_at >= 200000000LL;   /* (the opening's first updates are over) */
    if (!home_taken && cursor >= 0 && count > 0) {
        if (home != cursor) home = cursor, __atomic_store_n(&wheel_home, home, __ATOMIC_RELEASE);
        start_home = home;
        if (settled_in) home_taken = 1;
    }
    /* the pages: D-pad Left / ZL back, Right / ZR on; none of it reaches the game */
    const int pages = __atomic_load_n(&page_count, __ATOMIC_ACQUIRE);
    int page = __atomic_load_n(&wheel_page, __ATOMIC_ACQUIRE);
    if (page >= pages) page = pages > 0 ? pages - 1 : 0;
    const uint64_t pressed = buttons & ~prev_buttons;
    prev_buttons = buttons;
    int turn = (pressed & ((1ULL << B_RIGHT) | (1ULL << B_ZR))) ? 1 : (pressed & ((1ULL << B_LEFT) | (1ULL << B_ZL))) ? -1 : 0;
    buttons &= ~((1ULL << B_LEFT) | (1ULL << B_RIGHT) | (1ULL << B_UP) | (1ULL << B_DOWN) | (1ULL << B_ZL) | (1ULL << B_ZR));
    if (turn && pages > 1) {
        page = (page + turn + pages) % pages;
        stick_slot = -1, cued = -1, pointed = -1, mouse_live = 0, ref_x = mx, ref_y = my;
        hos_audio_cue("SE_Sys_ShortCut_CursolMove_00");
        if (wheel_trace()) hos_log("wheel: page %d of %d", page + 1, pages);
    }
    __atomic_store_n(&wheel_page, page, __ATOMIC_RELEASE);
    const uint32_t laid = __atomic_load_n(&page_mask, __ATOMIC_ACQUIRE);
    /* (a page just turned to: nothing to pick until laid out; then any slot, an empty one too: where a held item would go) */
    const uint32_t mask = (int)(laid >> 8) != page ? 0 : 0xff;
    const uint32_t items = (int)(laid >> 8) != page ? 0 : laid & 0xff;   /* (the slots that hold an item) */
    /* the stick */
    /* (either stick: the one pushed further; the camera does not see the right one while the wheel is open) */
    const double lx = stick[0], ly = stick[1], rx = stick[2], ry = stick[3];
    const int right = rx * rx + ry * ry > lx * lx + ly * ly;
    const double x = right ? rx : lx, y = right ? ry : ly, d2 = x * x + y * y;
    stick[2] = stick[3] = 0;
    if (d2 > 18000.0 * 18000.0) {
        int s = slot_toward(x, y, MKWVK_WHEEL_SLOTS);
        if (stick_slot >= 0 && s != stick_slot && slot_off(x, y, stick_slot, MKWVK_WHEEL_SLOTS) < 30.0) s = stick_slot;
        if (mask >> s & 1) stick_slot = s;
    } else if (d2 < 12000.0 * 12000.0) {
        stick_slot = -1;
    }
    stick[0] = stick[1] = 0;
    int pick = stick_slot;
    /* the mouse's pick: once the pointer has moved this opening, and since the stick last pointed (a pointer left on an
       item does not take the pick back from the stick) */
    if (pick >= 0) {
        ref_x = mx, ref_y = my, mouse_live = 0;
    } else if (mouse_ok) {
        if (!mouse_live && (abs(mx - ref_x) > 4 || abs(my - ref_y) > 4)) mouse_live = 1;
        if (mouse_live) {
            const int s = slot_under(mx, my, mw, mh, MKWVK_WHEEL_SLOTS);
            pick = s >= 0 && (mask >> s & 1) ? s : -1;
        }
    }
    if (clicked && !click_close) {
        const int s = slot_under(cx, cy, mw, mh, MKWVK_WHEEL_SLOTS);
        if (s >= 0 && (items >> s & 1) && moving < 0) {
            pick = s, click_close = 1, mouse_live = 1;
            if (wheel_trace()) hos_log("wheel: clicked slot %d of page %d: closing", s, page + 1);
        }
    }
    /* (the slot pointed at stays pointed at once the stick is let go: Y and the ring act on it) */
    if (pick >= 0) pointed = pick;
    pick = pointed;
    __atomic_store_n(&wheel_pick, pick, __ATOMIC_RELEASE);
    if (pick >= 0 && pick != cued) hos_audio_cue("SE_Sys_ShortCut_CursolMove_00");
    cued = pick >= 0 ? pick : cued;
    const int key = pick >= 0 ? page * MKWVK_WHEEL_SLOTS + pick : -1;
    /* Y: an item taken up to be moved, or the one held put down where the stick points (on its own slot: put back). The
       game never sees Y while the wheel is open */
    if (pressed & (1ULL << B_Y)) {
        if (moving < 0 && key >= 0 && (items >> pick & 1)) {
            moving = key, __atomic_store_n(&move_from, key, __ATOMIC_RELEASE);
            hos_audio_cue("SE_Sys_ShortCut_CursolMove_00");
            if (wheel_trace()) hos_log("wheel: slot %d of page %d taken up to be moved", pick, page + 1);
        } else if (moving >= 0 && key >= 0) {
            if (key != moving) {
                __atomic_store_n(&drop_req, moving << 16 | key, __ATOMIC_RELEASE);
                __atomic_add_fetch(&drop_seq, 1, __ATOMIC_ACQ_REL);
            }
            moving = -1, __atomic_store_n(&move_from, -1, __ATOMIC_RELEASE);
            hos_audio_cue("SE_Sys_ShortCut_CursolMove_00");
        }
    }
    /* B while one is held: put back where it was (the game does not see it then) */
    if (moving >= 0 && (pressed & (1ULL << B_B))) {
        moving = -1, __atomic_store_n(&move_from, -1, __ATOMIC_RELEASE);
        hos_audio_cue("SE_Sys_ShortCut_CursolMove_00");
    }
    if (moving >= 0 || (prev_moving_b && (buttons & (1ULL << B_B)))) buttons &= ~(1ULL << B_B);
    prev_moving_b = moving >= 0 || (prev_moving_b && (buttons & (1ULL << B_B)));
    buttons &= ~(1ULL << B_Y);
    /* (a choice only while the player holds R, or by a click: the slot pointed at stays pointed at once R is let go, and
       choosing it again after it was equipped kept R held for the game and the row open, its pick flickering) */
    if (key >= 0 && key != chosen && moving < 0 && (items >> pick & 1) && (r_pressed || clicked) && (!click_close || clicked)) {   /* (after a click, the clicked item stays chosen) */
        chosen = key;
        choose(key);
        if (wheel_trace()) hos_log("wheel: chose slot %d of page %d (cursor %d)", pick, page + 1, cursor);
        /* (a pick ends the survey and home with it) */
        if (home >= 0) home = -1, __atomic_store_n(&wheel_home, -1, __ATOMIC_RELEASE);
        home_taken = 1;
    }
    /* R held for the game until the row has reached the item and the last tap has been let go: a D-pad press that
       outlives the row reaches play, where LEFT opened the pause menu */
    const int answered = __atomic_load_n(&choice_ack, __ATOMIC_ACQUIRE) == __atomic_load_n(&choice_seq, __ATOMIC_ACQUIRE);
    const int lost = chosen >= 0 && answered && __atomic_load_n(&choice_lost, __ATOMIC_ACQUIRE);
    /* where the taps go: the item chosen (its place, or the next stop of the search for it); with none, the survey's next
       stop while it runs, else home */
    /* (an item the game knows by number: the row is left where it opened, its close equips nothing new, and the item is put
       on ZR at the HUD's next update, at once however far it is in the row) */
    const int instant = chosen >= 0 && answered && !lost && start_home >= 0 && __atomic_load_n(&choice_item, __ATOMIC_ACQUIRE) >= 0;
    int aim = -1;
    if (instant) {
        aim = start_home;
    } else if (chosen >= 0) {
        aim = answered && !lost ? __atomic_load_n(&choice_row, __ATOMIC_ACQUIRE) : -1;
    } else if (home >= 0) {
        const int stop = settled_in && r_pressed && !click_close && now < survey_until
                             ? survey_stop(cursor, count, __atomic_load_n(&row_known, __ATOMIC_ACQUIRE))
                             : -1;
        aim = stop >= 0 ? stop : home;
    }
    const int settled = (instant ? cursor == start_home
                         : chosen >= 0 ? answered && (lost || __atomic_load_n(&choice_here, __ATOMIC_ACQUIRE))
                                       : home < 0 || cursor == home) && !tap_dir;
    if (click_close) buttons &= ~HOLD_BUTTONS;   /* (a click closes: R let go for the game, as by the player) */
    int r_held = (buttons & HOLD_BUTTONS) != 0;
    if (!r_held) {
        if (!r_let_go_at) {
            r_let_go_at = now;
            if (!settled && wheel_trace()) hos_log("wheel: R let go before the row got there (cursor %d, aim %d): kept held", cursor, aim);
        }
        /* (at most 4 s: a 20-item row's far end is 19 taps of 0.12 s away) */
        if (!settled && now - r_let_go_at < 4000000000LL) r_held = 1, buttons |= 1ULL << B_R;
    } else {
        r_let_go_at = 0;
    }
    r_kept = r_held;
    if (!r_held && settled && chosen >= 0) {   /* (done: the next opening starts afresh, however soon it comes) */
        if (lost && wheel_trace()) hos_log("wheel: the item chosen is not in the row: nothing equipped");
        if (instant) {
            __atomic_store_n(&equip_item, __atomic_load_n(&choice_item, __ATOMIC_ACQUIRE), __ATOMIC_RELEASE);
            __atomic_add_fetch(&equip_seq, 1, __ATOMIC_ACQ_REL);
        }
        chosen = -1;
        choose(-1);
    }
    /* a tap toward the aim, only while R is held, each after the last was let go and the cursor had a frame to follow.
       The reads counted are the game's frames: a read within 5 ms of the last is the same frame's (it reads several pad
       styles) */
    const int new_frame = now - last_read > 5000000LL;
    if (new_frame) last_read = now;
    if (!r_held) tap_dir = 0, tap_down_n = 0;
    if (r_held && !tap_dir && tap_wait_n <= 0 && aim >= 0 && cursor >= 0 && cursor != aim) {
        tap_dir = aim > cursor ? 1 : -1;
        tap_down_n = tap_frames, tap_wait_n = 2 * tap_frames;
        if (wheel_trace())
            hos_log("wheel: tap %s (cursor %d, %s %d, at %lld ms)", tap_dir > 0 ? "RIGHT" : "LEFT", cursor,
                    chosen >= 0 ? "item" : aim == home ? "home" : "survey", aim, (long long)(now / 1000000));
    }
    if (tap_dir) buttons |= 1ULL << (tap_dir > 0 ? B_RIGHT : B_LEFT);
    if (new_frame) {
        if (tap_dir && --tap_down_n <= 0) tap_dir = 0;
        if (tap_wait_n > 0) tap_wait_n--;
    }
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
    const uint64_t screen = screen_arg;
    const uint64_t vt = peek64(screen);
    const int mine = vt == ITEM_CHOICE_VTABLE && wheel_enabled();
    /* (Cycle Items: play is the HUD updating without the pause menu; a tap is acted on at the HUD's update) */
    if (vt == CY_HUD_VTABLE) {
        __atomic_store_n(&hud_tick_ns, hos_now_ns(), __ATOMIC_RELEASE);
        static int cycle_seen;
        const int seq = __atomic_load_n(&cycle_seq, __ATOMIC_ACQUIRE);
        /* (under the wheel's lock: the row's places are the menu tick's; busy, the tap waits for the next frame) */
        static int equip_seen;
        const int eq = __atomic_load_n(&equip_seq, __ATOMIC_ACQUIRE);
        /* (the wheel's choice, once the row has closed: put on ZR while the row still showed, its old order was taken as the
           row's again) */
        const int row_shut = hos_now_ns() - __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE) > ROW_OPEN_NS;
        if (eq != equip_seen && row_shut && !__atomic_exchange_n(&state_lock, 1, __ATOMIC_ACQUIRE)) {
            equip_seen = eq;
            equip_now(c, __atomic_load_n(&equip_item, __ATOMIC_ACQUIRE));
            __atomic_store_n(&state_lock, 0, __ATOMIC_RELEASE);
        }
        if (seq != cycle_seen && !__atomic_exchange_n(&state_lock, 1, __ATOMIC_ACQUIRE)) {
            cycle_seen = seq;
            const int level = cycle_level();
            if (level > 0) cycle_now(c, __atomic_load_n(&cycle_dir, __ATOMIC_ACQUIRE), level);
            __atomic_store_n(&state_lock, 0, __ATOMIC_RELEASE);
        }
    } else if (vt == CY_PLUS_VTABLE || vt == CY_OVERLAY_VTABLE) {
        __atomic_store_n(&pause_tick_ns, hos_now_ns(), __ATOMIC_RELEASE);
    }
    if (vt != ITEM_CHOICE_VTABLE) __atomic_store_n(&other_tick_ns, hos_now_ns(), __ATOMIC_RELEASE);   /* (row_gone) */
    /* (the screens update on several worker threads at once: one at a time here, the others skip this frame) */
    if (__atomic_exchange_n(&state_lock, 1, __ATOMIC_ACQUIRE)) return mine;
    if (mine) {
        static uint64_t open_screen;
        const int64_t now = hos_now_ns();
        const int64_t last = __atomic_load_n(&row_tick_ns, __ATOMIC_ACQUIRE);
        /* (a new opening: another screen, a long gap, or the input side saw the row closed: reopened at once after an
           item was equipped, the gap is short, and the game had already put that item first) */
        const int closed = __atomic_exchange_n(&wheel_closed, 0, __ATOMIC_ACQ_REL);   /* (taken each update, used or not) */
        const int opened = screen != open_screen || !last || now - last > ROW_STALL_NS || closed;
        if (opened && wheel_trace())
            hos_log("wheel: the row opens (screen %#llx%s, %.0f ms after its last update)", (unsigned long long)screen,
                    screen != open_screen ? ", a new one" : "", last ? (double)(now - last) / 1e6 : -1.0);
        open_screen = screen;
        __atomic_store_n(&row_tick_ns, now, __ATOMIC_RELEASE);
        wheel_tick(c, screen, opened);
    }
    __atomic_store_n(&state_lock, 0, __ATOMIC_RELEASE);
    return mine;
}
