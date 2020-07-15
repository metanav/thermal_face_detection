%module gd3
%include "stdint.i"

%{
extern void init();
extern void load(uint8_t *img);
extern void display(float temperature, uint8_t detected, int8_t count);
%}

%typemap(in) (uint8_t *img) {
  $1 = (uint8_t *) PyBytes_AsString($input);
}

extern void init();
extern void load(uint8_t *img);
extern void display(float temperature, uint8_t detected, int8_t count);
