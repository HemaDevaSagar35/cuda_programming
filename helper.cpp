#include <stdio.h>
#include <iostream>
#include <random>

void generateRandomMarix(float* array, int size, int seed){
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int i = 0; i < size; i++){
        array[i] = dis(gen);
    }

    for(int i = 0;i < size;i++){
        std::cout << array[i] << std::endl;
    }
}

int main(){

    int size = 10;
    float* a = new float[size];
    float* b = new float[size];

    // std::random_device rd;
    generateRandomMarix(a, size, 11);
    generateRandomMarix(b, size, 21);
    return 0;

}