#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "EVE.h"

void init()
{
    EVE_Init();
}

void load(uint8_t *img)
{
    uint8_t buf[128];
    int8_t flag = 0;
    uint8_t i;

    EVE_LIB_BeginCoProList();
    EVE_CMD_LOADIMAGE(0, 0);
    
    while (flag != 2) {
    	for (i = 0; i < sizeof(buf); i++) {
    	    buf[i] = *img++;
    	    if (buf[i] == 0xff) {
    	    	flag = 1;
    	    } else {
    	    	if (flag == 1) {
    	    	    if (buf[i] == 0xd9) {
			flag = 2;
			i++;
			break;
		    }
    	    	}
		flag = 0;
    	    }
    	}
    	EVE_LIB_WriteDataToCMD(buf, (i + 3) & (~3));
    };
    EVE_LIB_EndCoProList();
    EVE_LIB_AwaitCoProEmpty();
}

void display(float temperature, int8_t count)
{
    uint32_t eve_addr;
    uint32_t img_width;
    uint32_t img_height;
    uint8_t scale = 2;
    char tem[20];
    char counter[6];

    sprintf(tem, "Temperature: %04.1f", temperature);
    EVE_LIB_GetProps(&eve_addr, &img_width, &img_height);

    EVE_LIB_BeginCoProList();
    EVE_CMD_DLSTART();
    EVE_BEGIN(EVE_BEGIN_BITMAPS);
    EVE_CLEAR_COLOR_RGB(0, 0, 0);
    EVE_CLEAR(1,1,1);
    EVE_COLOR_RGB(255, 255, 255);
    EVE_BITMAP_HANDLE(0);
    EVE_BITMAP_SOURCE(0);
    EVE_BITMAP_LAYOUT(EVE_FORMAT_RGB565, img_width * 2, img_height);
    EVE_BITMAP_LAYOUT_H((img_width * 2) >> 10, img_height >> 9);
    EVE_BITMAP_SIZE(EVE_FILTER_NEAREST, EVE_WRAP_BORDER, EVE_WRAP_BORDER, 
		    img_width * scale, img_height * scale);
    EVE_BITMAP_SIZE_H(img_width >> 9, img_height >> 9);
    EVE_BEGIN(EVE_BEGIN_BITMAPS);
    EVE_CMD_TEXT(0, 0, 28, 0, tem);
    EVE_CMD_TEXT(180, 0, 20, 0, "o");
    EVE_CMD_TEXT(182, 0, 28, 0, " C");
    EVE_VERTEX2II(0, 30, 0, 0);

    if (count > -1) {
        EVE_COLOR_RGB(255, 0, 0);
        EVE_CMD_TEXT(245, 30, 29, 0, "High temperature!");
        EVE_COLOR_RGB(255, 225, 255);
        EVE_CMD_TEXT(245, 62, 29, 0, "Scan the QR Code to");
        EVE_CMD_TEXT(245, 90, 29, 0, "know more.");
	sprintf(counter, "00:%02d", count);
        EVE_CMD_TEXT(290, 130, 31, 0, counter);
    }

    EVE_END();
    EVE_DISPLAY();
    EVE_CMD_SWAP();
    EVE_LIB_EndCoProList();
    EVE_LIB_AwaitCoProEmpty();
}
