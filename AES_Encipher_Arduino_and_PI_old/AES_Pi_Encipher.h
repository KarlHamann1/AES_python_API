#pragma once

#include <stdint.h>

// Uncomment if 128-bit (AES128), 192-bit (AES192), or 256-bit (AES256).
#define AES128 1
//#define AES192 1
//#define AES256 1

// Hard-coded key bytes (same as Arduino code)
#define KEY_BYTE_0   129
#define KEY_BYTE_1   85
#define KEY_BYTE_2   100
#define KEY_BYTE_3   7
#define KEY_BYTE_4   226
#define KEY_BYTE_5   183
#define KEY_BYTE_6   192
#define KEY_BYTE_7   144
#define KEY_BYTE_8   210
#define KEY_BYTE_9   49
#define KEY_BYTE_10  70
#define KEY_BYTE_11  180
#define KEY_BYTE_12  108
#define KEY_BYTE_13  213
#define KEY_BYTE_14  84
#define KEY_BYTE_15  162

#ifdef __cplusplus
extern "C" {
#endif

// AES block is always 16 bytes
#define AES_BLOCKLEN 16 

// Key-length logic
#if defined(AES256) && (AES256 == 1)
    #define AES_KEYLEN      32
    #define AES_keyExpSize  240
    #define Nk              8
    #define Nr              14
#elif defined(AES192) && (AES192 == 1)
    #define AES_KEYLEN      24
    #define AES_keyExpSize  208
    #define Nk              6
    #define Nr              12
#else  // AES128
    #define AES_KEYLEN      16
    #define AES_keyExpSize  176
    #define Nk              4
    #define Nr              10
#endif

// A simple 4x4 state matrix
typedef uint8_t state_t[4][4];

// Functions to call externally
void aes_init(void);                             // sets up key schedule
void aes_set_state(const uint8_t *plaintext16);  // sets the 16-byte state
void aes_get_state(uint8_t *out16);              // reads the 16-byte state
void aes_encrypt(void);                          // runs AES on current state (toggles GPIO before/after)

#ifdef __cplusplus
}
#endif
