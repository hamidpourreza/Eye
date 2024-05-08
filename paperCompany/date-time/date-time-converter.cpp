//
// Created by mj on 09.07.23.
//

#include "date-time-converter.h"
long long gregorianToJalali(long gy, long gm, long gd) {
    long days;
    long jm, jd;
    {
        long gy2 = (gm > 2) ? (gy + 1) : gy;
        long g_d_m[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
        days = 355666 + (365 * gy) + ((int) ((gy2 + 3) / 4)) - ((int) ((gy2 + 99) / 100)) +
               ((int) ((gy2 + 399) / 400)) + gd + g_d_m[gm - 1];
    }
    long jy = -1595 + (33 * ((int) (days / 12053)));
    days %= 12053;
    jy += 4 * ((int) (days / 1461));
    days %= 1461;
    if (days > 365) {
        jy += (int) ((days - 1) / 365);
        days = (days - 1) % 365;
    }
    if (days < 186) {
        jm = 1 + (int) (days / 31);
        jd = 1 + (days % 31);
    } else {
        jm = 7 + (int) ((days - 186) / 30);
        jd = 1 + ((days - 186) % 30);
    }
    long long time = jy * 100'000'000 + jm * 1'000'000 + jd * 10'000;
    return time;
}