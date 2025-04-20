#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  
#include <time.h>
#include "nn.h"

void load_mnist()
{
    FILE *training_images_file = fopen("../MNIST_bin/mnist_train_images.bin", "rb");
    if (training_images_file == NULL)
    {
        printf("Error opening training images file\n");
        exit(1);
    }

    FILE *training_labels_file = fopen("../MNIST_bin/mnist_train_labels.bin", "rb");
    if (training_labels_file == NULL)
    {
        printf("Error opening training labels file\n");
        exit(1);
    }

    FILE *test_images_file = fopen("../MNIST_bin/mnist_test_images.bin", "rb");
    if (test_images_file == NULL)
    {
        printf("Error opening test images file\n");
        exit(1);
    }

    FILE *test_labels_file = fopen("../MNIST_bin/mnist_test_labels.bin", "rb");
    if (test_labels_file == NULL)
    {
        printf("Error opening test labels file\n");
        exit(1);
    }

    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }

    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                training_labels[i][j] = 1;
            }
            else
            {
                training_labels[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, test_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                test_labels[i][j] = 1;
            }
            else
            {
                test_labels[i][j] = 0;
            }
        }
    }

    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
}

int max_index(double arr[], int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

void save_weights_biases(const char* filename)
{
    FILE* file = fopen(filename, "wb");
    if (file == NULL)
    {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    
    fwrite(weight1, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
    fwrite(bias2, sizeof(double), OUTPUT_NODES, file);
    
    fclose(file);
    printf("Weights and biases saved to %s\n", filename);
}
