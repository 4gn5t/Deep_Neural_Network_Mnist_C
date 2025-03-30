#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>  
#include <string.h>

#define INPUT_NODES 784  
#define HIDDEN_NODES 256 
#define OUTPUT_NODES 10  

#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000


#define NUMBER_OF_EPOCHS 10

double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

void load_mnist()
{
    FILE *training_images_file = NULL;
    FILE *training_labels_file = NULL;
    FILE *test_images_file = NULL;
    FILE *test_labels_file = NULL;
    
    
    training_images_file = fopen("mnist_train_images.bin", "rb");
    training_labels_file = fopen("mnist_train_labels.bin", "rb");
    test_images_file = fopen("mnist_test_images.bin", "rb");
    test_labels_file = fopen("mnist_test_labels.bin", "rb");
    
    printf("All MNIST files opened successfully\n");
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            
            for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
            {
                unsigned char pixels[INPUT_NODES];
                
                fread(pixels, sizeof(unsigned char), INPUT_NODES, training_images_file);
                
                
                for (int j = 0; j < INPUT_NODES; j++)
                {
                    training_images[i][j] = (double)pixels[j] / 255.0;
                }
            }
            printf("Thread %d: Training images loaded\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            
            for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
            {
                unsigned char label;
                fread(&label, sizeof(unsigned char), 1, training_labels_file);
                for (int j = 0; j < OUTPUT_NODES; j++)
                {
                    training_labels[i][j] = (j == label) ? 1.0 : 0.0;
                }
            }
            printf("Thread %d: Training labels loaded\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            
            for (int i = 0; i < NUM_TEST_IMAGES; i++)
            {
                unsigned char pixels[INPUT_NODES];
                fread(pixels, sizeof(unsigned char), INPUT_NODES, test_images_file);
                
                for (int j = 0; j < INPUT_NODES; j++)
                {
                    test_images[i][j] = (double)pixels[j] / 255.0;
                }
            }
            printf("Thread %d: Test images loaded\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            
            for (int i = 0; i < NUM_TEST_IMAGES; i++)
            {
                unsigned char label;
                fread(&label, sizeof(unsigned char), 1, test_labels_file);
                for (int j = 0; j < OUTPUT_NODES; j++)
                {
                    test_labels[i][j] = (j == label) ? 1.0 : 0.0;
                }
            }
            printf("Thread %d: Test labels loaded\n", omp_get_thread_num());
        }
    }
    
    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
    
    printf("MNIST data loaded successfully\n");
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
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

void feedforward(double input[INPUT_NODES], 
                double weight1[INPUT_NODES][HIDDEN_NODES],
                double weight2[HIDDEN_NODES][OUTPUT_NODES],
                double bias1[HIDDEN_NODES],
                double bias2[OUTPUT_NODES],
                double hidden[HIDDEN_NODES],
                double output_layer[OUTPUT_NODES]) 
{
    
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j][i];
        }
        sum += bias1[i];
        hidden[i] = sigmoid(sum);
    }
    
    
    
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[j] * weight2[j][i];
        }
        sum += bias2[i];
        output_layer[i] = sigmoid(sum);
    }
}

void train(double input[INPUT_NODES], double output[OUTPUT_NODES], 
    double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], 
    double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], 
    int correct_label, int *correct_counter)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    
    feedforward(input, weight1, weight2, bias1, bias2, hidden, output_layer);

    int index = max_index(output_layer, OUTPUT_NODES);
    
    
    if (index == correct_label) {
        #pragma omp atomic
        (*correct_counter)++;
    }
    
    
    double delta_output[OUTPUT_NODES];
    
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        delta_output[i] = (output[i] - output_layer[i]) * output_layer[i] * (1 - output_layer[i]);
    }
    
    double delta_hidden[HIDDEN_NODES];
    
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        delta_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            delta_hidden[i] += delta_output[j] * weight2[i][j];
        }
        delta_hidden[i] *= hidden[i] * (1 - hidden[i]);
    }
    
    
    double learning_rate = 0.1;
    
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] += learning_rate * delta_output[j] * hidden[i];
        }
    }
    
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] += learning_rate * delta_hidden[j] * input[i];
        }
    }
    
    
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] += learning_rate * delta_hidden[i];
    }
    
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] += learning_rate * delta_output[i];
    }
    
    
    #pragma omp flush(weight1, weight2, bias1, bias2)
}

double init_nodes_weight()
{
    
    double xavier_init_hidden = sqrt(6.0 / (INPUT_NODES + HIDDEN_NODES));
    double xavier_init_output = sqrt(6.0 / (HIDDEN_NODES + OUTPUT_NODES));
    
    
    #pragma omp parallel
    {
        
        #pragma omp master
        {
            printf("Weights initialization started by master thread %d\n", omp_get_thread_num());
        }
        
        #pragma omp barrier
        
        #pragma omp sections
        {
            #pragma omp section
            {
                printf("Thread %d initializing input-hidden weights\n", 
                       omp_get_thread_num());
                for (int i = 0; i < INPUT_NODES; i++) {
                    for (int j = 0; j < HIDDEN_NODES; j++) {
                        weight1[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier_init_hidden;
                    }
                }
            }
            
            #pragma omp section
            {
                printf("Thread %d initializing hidden-output weights\n", 
                       omp_get_thread_num());
                for (int i = 0; i < HIDDEN_NODES; i++) {
                    for (int j = 0; j < OUTPUT_NODES; j++) {
                        weight2[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier_init_output;
                    }
                }
            }
            
            #pragma omp section
            {
                printf("Thread %d initializing biases\n", 
                       omp_get_thread_num());
                
                for (int i = 0; i < HIDDEN_NODES; i++) {
                    bias1[i] = 0.0;
                }
                
                for (int i = 0; i < OUTPUT_NODES; i++) {
                    bias2[i] = 0.0;
                }
            }
        } 
        
        
        #pragma omp single
        {
            printf("Weight initialization completed by thread %d\n", 
                   omp_get_thread_num());
        }
    } 
}

void test(double input[INPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES], 
    double weight2[HIDDEN_NODES][OUTPUT_NODES], double bias1[HIDDEN_NODES], 
    double bias2[OUTPUT_NODES], int correct_label, int *correct_counter)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    
    feedforward(input, weight1, weight2, bias1, bias2, hidden, output_layer);
    
    int index = max_index(output_layer, OUTPUT_NODES);
    
    if (index == correct_label) {
        (*correct_counter)++;
    }
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

void task_based_training(int epoch) {
    int correct_train = 0;
    printf("Starting task-based training for epoch %d\n", epoch);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Generating training tasks on thread %d\n", omp_get_thread_num());
            
            
            for (int chunk = 0; chunk < NUM_TRAINING_IMAGES; chunk += 1000) {
                #pragma omp task
                {
                    int local_correct = 0;
                    int end = chunk + 1000;
                    if (end > NUM_TRAINING_IMAGES) end = NUM_TRAINING_IMAGES;
                    
                    printf("Thread %d processing chunk %d to %d\n", 
                           omp_get_thread_num(), chunk, end-1);
                    
                    for (int i = chunk; i < end; i++) {
                        int correct_label = max_index(training_labels[i], OUTPUT_NODES);
                        train(training_images[i], training_labels[i], weight1, weight2, 
                              bias1, bias2, correct_label, &local_correct);
                    }
                    
                    #pragma omp atomic
                    correct_train += local_correct;
                }
            }
            
            
            #pragma omp taskwait
            printf("All training tasks completed for epoch %d\n", epoch);
            printf("Epoch %d : Training Accuracy: %f\n", 
                   epoch, (double)correct_train / NUM_TRAINING_IMAGES);
        }
    }
}


void taskloop_based_testing() {
    int correct_predictions = 0;
    printf("Starting taskloop-based testing...\n");
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            
            #pragma omp taskloop grainsize(500) reduction(+:correct_predictions)
            for (int i = 0; i < NUM_TEST_IMAGES; i++) {
                int correct_label = max_index(test_labels[i], OUTPUT_NODES);
                int local_correct = 0;
                test(test_images[i], weight1, weight2, bias1, bias2, 
                     correct_label, &local_correct);
                correct_predictions += local_correct;
                
                if ((i+1) % 1000 == 0) {
                    printf("Thread %d completed testing image %d\n", 
                           omp_get_thread_num(), i);
                }
            }
        }
    }
    
    printf("Taskloop testing completed. Accuracy: %f\n", 
           (double)correct_predictions / NUM_TEST_IMAGES);
}


void ordered_batch_training(int batch_start, int batch_size) {
    printf("Starting ordered batch training from index %d\n", batch_start);
    double batch_time = omp_get_wtime();
    
    
    #pragma omp parallel for ordered schedule(dynamic)
    for (int i = batch_start; i < batch_start + batch_size; i++) {
        if (i < NUM_TRAINING_IMAGES) {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            int correct_counter = 0;
            
            train(training_images[i], training_labels[i], weight1, weight2, 
                  bias1, bias2, correct_label, &correct_counter);
            
            
            #pragma omp ordered
            {
                if ((i+1) % 1000 == 0) {
                    printf("Thread %d processed image %d\n", 
                           omp_get_thread_num(), i);
                }
            }
        }
    }
    
    double elapsed = omp_get_wtime() - batch_time;
    printf("Batch training completed in %.2f seconds\n", elapsed);
}


void lock_based_evaluation(int start_idx, int count) {
    omp_lock_t evaluation_lock;
    omp_init_lock(&evaluation_lock);
    
    int local_correct = 0;
    
    #pragma omp parallel
    {
        int thread_correct = 0;
        
        #pragma omp for nowait
        for (int i = start_idx; i < start_idx + count; i++) {
            if (i < NUM_TEST_IMAGES) {
                int correct_label = max_index(test_labels[i], OUTPUT_NODES);
                test(test_images[i], weight1, weight2, bias1, bias2, 
                     correct_label, &thread_correct);
            }
        }
        
        
        omp_set_lock(&evaluation_lock);
        local_correct += thread_correct;
        omp_unset_lock(&evaluation_lock);
        
    
        #pragma omp barrier
    
        #pragma omp single
        {
            printf("Lock-based evaluation completed. Correct: %d/%d\n", 
                   local_correct, count);
        }
    }
    
    omp_destroy_lock(&evaluation_lock);
}


int main()
{
    int correct_predictions = 0;
    
    
    srand(42);     
    
    int num_threads = omp_get_max_threads();
    printf("Running with %d OpenMP threads\n", num_threads);
    
    double start_time = omp_get_wtime();
    
    init_nodes_weight();

    double init_time = omp_get_wtime();
    load_mnist();
    double load_time = omp_get_wtime();
    printf("MNIST data loaded in %.2f seconds\n", load_time - init_time);

    printf("Starting training...\n");
    
    
    for(int epoch=0; epoch < NUMBER_OF_EPOCHS/2; epoch++)
    {
        int correct_train = 0;
        
        
        for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train(training_images[i], training_labels[i], weight1, weight2, bias1, bias2, correct_label, &correct_train);
            
            if ((i+1) % 10000 == 0) {
                printf("Progress: %d/%d images processed in epoch %d\n", i+1, NUM_TRAINING_IMAGES, epoch);
            }
        }
        printf("Epoch %d : Training Accuracy: %f\n", epoch, (double)correct_train / NUM_TRAINING_IMAGES);
    }
    
    
    for(int epoch=NUMBER_OF_EPOCHS/2; epoch < NUMBER_OF_EPOCHS; epoch++)
    {
        
        task_based_training(epoch);
        
        
        if (epoch % 10 == 0) {
            ordered_batch_training(0, 5000);
        }
    }
    
    save_weights_biases("model.bin");
    
    double train_end_time = omp_get_wtime();
    printf("Training completed in %.2f seconds\n", train_end_time - start_time);

    
    char choice;
    printf("\nDo you want to test specific images? (y/n): ");
    scanf(" %c", &choice);
    
    if (choice == 'y' || choice == 'Y') {
        while (1) {
            int image_index;
            printf("Enter image index (0-%d) or -1 to exit: ", NUM_TEST_IMAGES-1);
            scanf("%d", &image_index);
            
            if (image_index == -1) {
                break;
            }
            
            if (image_index < 0 || image_index >= NUM_TEST_IMAGES) {
                printf("Invalid image index. Please try again.\n");
                continue;
            }
            
            
            
            double hidden[HIDDEN_NODES];
            double output_layer[OUTPUT_NODES];
            
            
            feedforward(test_images[image_index], weight1, weight2, bias1, bias2, hidden, output_layer);
            
            int prediction = max_index(output_layer, OUTPUT_NODES);
            int true_label = max_index(test_labels[image_index], OUTPUT_NODES);
            
            printf("Prediction: %d\n", prediction);
            printf("True label: %d\n", true_label);
            printf("Result: %s\n\n", prediction == true_label ? "CORRECT" : "INCORRECT");
            
            printf("Confidence scores:\n");
            for (int i = 0; i < OUTPUT_NODES; i++) {
                printf("Digit %d: %.4f\n", i, output_layer[i]);
            }
            printf("\n");
        }
    }

    
    correct_predictions = 0;
    
    
    printf("Testing the network on %d images...\n", NUM_TEST_IMAGES);
    
    double test_start_time = omp_get_wtime();
    
    
    correct_predictions = 0;
    int half_test = NUM_TEST_IMAGES / 2;
    
    
    lock_based_evaluation(0, 1000);
    
    for (int i = 0; i < half_test; i++)
    {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], weight1, weight2, bias1, bias2, correct_label, &correct_predictions);
        
        if ((i+1) % 1000 == 0) {
            printf("Regular testing: %d/%d images. Accuracy: %f\n", 
                   i+1, half_test, (double)correct_predictions / (i+1));
        }
    }
    
    printf("Regular testing accuracy: %f\n", (double)correct_predictions / half_test);
    
    printf("Starting taskloop-based testing for remaining images...\n");
    taskloop_based_testing();
    
    double test_end_time = omp_get_wtime();
    printf("Testing completed in %.2f seconds\n", test_end_time - test_start_time);
    printf("Total execution time: %.2f seconds\n", test_end_time - start_time);
    printf("Correct predictions: %d/%d\n", correct_predictions, NUM_TEST_IMAGES);
    printf("Total test images: %d\n", NUM_TEST_IMAGES);
    
    return 0;
}

