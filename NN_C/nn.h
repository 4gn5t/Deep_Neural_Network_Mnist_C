#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  
#include <time.h>

#define INPUT_NODES 784  
#define HIDDEN_NODES 256 
#define OUTPUT_NODES 10  

#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000

#define NUMBER_OF_EPOCHS 10

extern double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
extern double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
extern double test_images[NUM_TEST_IMAGES][INPUT_NODES];
extern double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

extern double weight1[INPUT_NODES][HIDDEN_NODES];
extern double weight2[HIDDEN_NODES][OUTPUT_NODES];
extern double bias1[HIDDEN_NODES];
extern double bias2[OUTPUT_NODES];

int max_index(double arr[], int size);
void load_mnist();
void save_weights_biases(const char* filename);