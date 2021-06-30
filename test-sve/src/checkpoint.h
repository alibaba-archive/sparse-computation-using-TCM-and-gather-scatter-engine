// #include "m5ops.h"
# ifndef CHECKPOINT_H
# define CHECKPOINT_H


static void reset_stats(){
    /* resetstats before region of interest */
__asm__ __volatile__ (
"mov x0, #0; mov x1, #0; .inst 0XFF000110 | (0x40 << 16);" : : : "x0", "x1");
}

static void dump_stats(){
    /* dumpstats after region of interest */
__asm__ __volatile__ (
"mov x0, #0; mov x1, #0; .inst 0xFF000110 | (0x41 << 16);" : : : "x0", "x1");
}



#endif
