#include "aes.h"
#include "mini_uart.h"
#include "gpio.h"
#include "utils.h"

void kernel_main(void)
{
    uart_init();
    uart_send_string("Bare-Metal AES Test\r\n");

    // 1) Initialize AES
    aes_init();
    uart_send_string("AES init done.\r\n");

    // 2) Suppose we read 16 bytes from UART as plaintext
    //    For a simple test, let's define them statically:
    uint8_t plaintext[16] ={0x00,0x11,0x22,0x33,
                            0x44,0x55,0x66,0x77,
                            0x88,0x99,0xaa,0xbb,
                            0xcc,0xdd,0xee,0xff};

    aes_set_state(plaintext);

    // 3) Encrypt
    uart_send_string("Encrypting...\r\n");
    aes_encrypt();

    // 4) Read ciphertext
    uint8_t ciphertext[16];
    aes_get_state(ciphertext);

    // 5) Print ciphertext over UART
    uart_send_string("Ciphertext: ");
    for(int i=0; i<16; i++) {
        // For each byte, convert to hex
        // Example: 'A' = 0x41
        static const char *hex = "0123456789ABCDEF";
        uart_send(hex[(ciphertext[i] >> 4) & 0xF]);
        uart_send(hex[(ciphertext[i]     ) & 0xF]);
    }
    uart_send_string("\r\n");

    while (1) {
        // Echo loop or do more encryption cycles
        // ...
    }
}
