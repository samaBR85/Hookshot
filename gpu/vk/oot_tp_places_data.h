/* The Teleport places' data made from the player's dump at build time (nx2/teleport_places.py writes teleport_places.bin in
   the game folder): each place's position, arrival event and whether only warps and scenes reach it. Read by the game side
   (host/hos_main.c: the map change, which place a screenshot is of) and the menu (gpu/vk/vk_menu.cpp: the story tag), each
   once, C and C++. Without the file (an older build) every place arrives with the plain event, and no screenshot is taken. */
#pragma once
#include <stdio.h>
#include <string.h>
#include "oot_tp_places.h"

typedef struct {
    float pos[3];
    char event[64];   /* (the arrival event: "" when the file did not have the place) */
    int story;
} OotTpPlaceData;

/* data[OOT_TP_PLACES] from path ('TPP2', u32 count, then {char key[48]; float x, y, z; char event[64]; u8 story; u8 pad[3]}),
   matched by key; the number of places it had */
static int oot_tp_places_read(const char *path, OotTpPlaceData *data) {
    memset(data, 0, sizeof(OotTpPlaceData) * OOT_TP_PLACES);
    FILE *f = path && path[0] ? fopen(path, "rb") : NULL;
    if (!f) return 0;
    char head[8];
    int found = 0;
    if (fread(head, 1, 8, f) == 8 && !memcmp(head, "TPP2", 4)) {
        unsigned count;
        memcpy(&count, head + 4, 4);
        for (unsigned n = 0; n < count && n < 4096; n++) {
            unsigned char e[128];
            if (fread(e, 1, sizeof e, f) != sizeof e) break;
            e[47] = 0, e[123] = 0;
            for (int i = 0; i < OOT_TP_PLACES; i++)
                if (!strcmp(oot_tp_places[i].key, (const char *)e)) {
                    memcpy(data[i].pos, e + 48, 12);
                    memcpy(data[i].event, e + 60, 64);
                    data[i].story = e[124] != 0;
                    found++;
                    break;
                }
        }
    }
    fclose(f);
    return found;
}
