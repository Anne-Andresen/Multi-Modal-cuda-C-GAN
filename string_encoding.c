#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define VOCABULARY "abcdefghijklmnopqrstuvwxyz,._ -1234567890"
#define MAX_LENGTH 256

typedef struct {
    char *data;
    int length;
} string_t;

typedef struct {
    int *data;
    int length;
} array_t;

string_t string_concat(string_t a, string_t b) {
    string_t result;
    result.data = (char *)malloc((a.length + b.length + 1) * sizeof(char));
    strcpy(result.data, a.data);
    strcat(result.data, b.data);
    result.length = a.length + b.length;
    return result;
}

array_t string_to_array(string_t s) {
    array_t result;
    result.data = (int *)malloc(s.length * sizeof(int));
    for (int i = 0; i < s.length; i++) {
        result.data[i] = (int)s.data[i];
    }
    result.length = s.length;
    return result;
}

array_t encode_string(string_t s, string_t vocabulary) {
    array_t result;
    result.data = (int *)malloc(s.length * sizeof(int));
    for (int i = 0; i < s.length; i++) {
        char c = s.data[i];
        int index = -1;
        for (int j = 0; j < vocabulary.length; j++) {
            if (vocabulary.data[j] == c) {
                index = j;
                break;
            }
        }
        result.data[i] = index;
    }
    result.length = s.length;
    return result;
}

array_t repeat_array(array_t a, int repeat_factor) {
    array_t result;
    result.data = (int *)malloc((a.length * repeat_factor) * sizeof(int));
    for (int i = 0; i < repeat_factor; i++) {
        for (int j = 0; j < a.length; j++) {
            result.data[i * a.length + j] = a.data[j];
        }
    }
    result.length = a.length * repeat_factor;
    return result;
}

int main() {
    string_t text = {"hello", 5};
    string_t dose = {"1", 1};
    string_t collective_string = string_concat(text, dose);
    string_t vocabulary = {VOCABULARY, strlen(VOCABULARY)};
    array_t encoded = encode_string(collective_string, vocabulary);
    array_t data = encoded;
    for (int i = 0; i < data.length; i++) {
        data.data[i] /= 10;
    }
    int repeat_factor = 2097152 / data.length;
    array_t repeat_tensor = repeat_array(data, repeat_factor + 1);
    repeat_tensor.length -= data.length;
    // Reshape the array to 5D
    int dims[] = {1, 1, 128, 128, 128};
    int size = 1;
    for (int i = 0; i < 5; i++) {
        size *= dims[i];
    }
    array_t result;
    result.data = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        result.data[i] = repeat_tensor.data[i % repeat_tensor.length];
    }
    result.length = size;
    // Print result
    for (int i = 0; i < result.length; i++) {
        printf("%d ", result.data[i]);
    }
    return 0;
}s