// Frame interpolation (vk_framegen.cpp): made frames shown between the game's (in a window; RX_VK_FRAMEGEN=0: off).
#pragma once

#include "vk_internal.h"

namespace mkwvk {

bool framegen_on();
/// Records the frame's processing into the present's recording (src: the window texture's view, sw x sh, the crop
/// x, y, w, h); false when interpolation cannot run (the present goes to the window as without it)
bool framegen_record(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4]);
/// (hookshot) Who made the made frames of the sets shown lately (the last 64, about a second at 60): what the Insert menu's
/// Frame generation by row and the FPS counter name. None: none made (Off, nothing to blend from, none for half a second)
enum class FgMaker { None, BuiltIn, Dlss, Fsr3 };
FgMaker framegen_maker();
/// The first frame the last framegen_record made (null when it made none), w x h, GENERAL
VkImage framegen_made_image(uint32_t* w, uint32_t* h);
/// After that recording's submit (value): the display thread shows its frames
void framegen_queue(uint64_t value);
/// (hookshot) The Gameplay tab's Item wheel row is on (off by default) and the wheel's backdrop wanted
/// (RX_VK_WHEEL_BACKDROP=0: off). While it is false the two below record nothing and make nothing
bool framegen_wheel_on();
/// (hookshot) With the Item wheel row on: the gameplay upscale's output (image, its view format, w x h at level, GENERAL)
/// as the upscale left it, the scene before the game draws its HUD on it, copied in the frame's own work (the game stops
/// drawing its scene while its item row is shown: the frame before the row opens is the backdrop)
void framegen_wheel_scene(VkCommandBuffer cmd, VkImage image, VkFormat format, uint32_t w, uint32_t h, uint32_t level);
/// (hookshot) While the item wheel is open: the window texture's crop (dst, GENERAL, the game's RGBA8 picture) covered with
/// the scene framegen_wheel_scene last copied, blurred and dimmed, so the game's own item row and HUD don't show
/// behind the wheel. False (nothing recorded) when the wheel is closed, no scene was copied since the row was turned on,
/// or the picture is not that scene's
bool framegen_wheel_backdrop(VkCommandBuffer cmd, VkImage dst, VkFormat format, uint32_t sw, uint32_t sh, const int crop[4]);
/// The window's present of an image (GENERAL, w x h; crop as above or null), vk_replay.cpp: the display thread's
void present_image(VkImage src, uint32_t w, uint32_t h, const int crop[4], VkImageView view = VK_NULL_HANDLE);
/// (hookshot) Present ahead (vk_framegen.cpp display_main, #1702): while present_defer_into is set on a thread, present_image
/// submits the frame's copy into its swapchain image and stops there, with what present_commit needs to show it later. The
/// copy is then on the queue before the game's next frame, so its present at its time does not wait for that frame's rendering
struct PendingPresent {
    bool used = false;
    uint32_t index = 0;
    VkFence done = VK_NULL_HANDLE;           // (the copy's fence)
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint64_t reflex_frame = ~0ull;           // (NVIDIA Reflex: the frame it shows, vk_reflex.h reflex_present_frame)
    uint64_t antilag_frame = ~0ull;          // (AMD Anti-Lag 2: the frame it shows, vk_antilag.h antilag_present_frame)
    uint64_t swapchain_generation = 0;
};
extern thread_local PendingPresent* present_defer_into;
bool present_commit(const PendingPresent& p);   // (false: not shown, nothing deferred or a swapchain made again since)
/// The Insert menu's upscaling of a frame to the window (dw x dh), recorded: the image to show, or null when it is off.
/// pq: src is the game's own HDR10 picture (Native HDR), scaled into an RGB10A2 image
VkImage framegen_scale(VkCommandBuffer cmd, VkImageView src, VkImage image, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                       bool pq = false);
VkImageView framegen_scaled_view();   // (hookshot) the view of framegen_scale's last image
/// (hookshot) The HDR10 picture of src's crop at dw x dh (the Insert menu's HDR), GENERAL; null: none (the SDR picture then;
/// Native's scaled picture, else black: present_image).
/// pq: src is the game's own HDR10 picture (Native), its PQ codes as they are at 100% contrast, else graded and written again;
/// else the SDR picture, expanded
VkImage framegen_hdr(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                     bool pq = false);
/// Each present: ultrawide's menus (the Insert menu's setting to the game's side, the sides blurred or black while a menu shows)
void wide_sides_tick();
/// Whether ultrawide's menu sides can show at all: Ultrawide with a margin and the Insert menu's setting on
bool wide_sides_possible();

} // namespace mkwvk
