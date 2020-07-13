#ifndef _EVE_CONFIG_H_
#define _EVE_CONFIG_H_

#define FT800 1
#define FT801 2
#define FT811 3
#define FT812 4

#define WVGA 1

#ifndef FT8XX_TYPE
#define FT8XX_TYPE FT812
#endif

#ifndef FT8XX_DISPLAY
#define FT8XX_DISPLAY WVGA
#endif

#undef FT81X_ENABLE
#if (FT8XX_TYPE == FT811) || (FT8XX_TYPE == FT812)
#define FT81X_ENABLE
#endif

#if FT8XX_DISPLAY == WVGA
#define EVE_DISP_WIDTH 480 // Active width of LCD display
#define EVE_DISP_HEIGHT 272 // Active height of LCD display
#define EVE_DISP_HCYCLE 548 // Total number of clocks per line
#define EVE_DISP_HOFFSET 43 // Start of active line
#define EVE_DISP_HSYNC0 0 // Start of horizontal sync pulse
#define EVE_DISP_HSYNC1 41 // End of horizontal sync pulse
#define EVE_DISP_VCYCLE 292 // Total number of lines per screen
#define EVE_DISP_VOFFSET 12 // Start of active screen
#define EVE_DISP_VSYNC0 0 // Start of vertical sync pulse
#define EVE_DISP_VSYNC1 10 // End of vertical sync pulse
#define EVE_DISP_PCLK 5 // Pixel Clock
#define EVE_DISP_SWIZZLE 3 //prev=2  Define RGB output pins
#define EVE_DISP_PCLKPOL 1 // Define active edge of PCLK
#endif // FT81X_WQVGA

#endif /* _EVE_CONFIG_H_ */
