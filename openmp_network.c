#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>  
#include <string.h>

#define INPUT_NODES 784  // Кількість вхідних нейронів (28x28 пікселів)
#define HIDDEN_NODES 256 // Кількість нейронів прихованого шару
#define OUTPUT_NODES 10  // Кількість вихідних нейронів (цифри 0-9)

#define NUM_TRAINING_IMAGES 60000 // Загальна кількість тренувальних зображень
#define NUM_TEST_IMAGES 10000     // Загальна кількість тестових зображень

#define NUMBER_OF_EPOCHS 10 // Кількість епох навчання

// Масиви для зберігання даних MNIST
double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];   // Тренувальні зображення
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];  // Мітки тренувальних зображень
double test_images[NUM_TEST_IMAGES][INPUT_NODES];           // Тестові зображення
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];          // Мітки тестових зображень

// Ваги та зміщення нейронної мережі
double weight1[INPUT_NODES][HIDDEN_NODES];   // Ваги між вхідним і прихованим шарами
double weight2[HIDDEN_NODES][OUTPUT_NODES];  // Ваги між прихованим і вихідним шарами
double bias1[HIDDEN_NODES];                  // Зміщення прихованого шару
double bias2[OUTPUT_NODES];                  // Зміщення вихідного шару

// Функція для завантаження MNIST даних
// Відкриває файли з даними MNIST, читає їх у масиви та нормалізує пікселі зображень
// до діапазону [0, 1]. Також перетворює мітки в one-hot кодування.
// Використовує OpenMP для паралельного завантаження даних.
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
    
    // Створення паралельних секцій для завантаження зображень
    // Розподіл на 4 секції: 2 для зображень, 2 для міток
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Секція 1: Завантаження тренувальних зображень
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
            // Секція 2: Завантаження тренувальних міток
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
            // Секція 3: Завантаження тестових зображень
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
            // Секція 4: Завантаження тестових міток
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

// Функція для обчислення сигмоїдної функції
// Використовується для обчислення активації нейронів у прихованому та вихідному шарах
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Функція для обчислення індексу максимального значення в масиві
// Використовується для визначення класу, до якого належить зображення
int max_index(double arr[], int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

// Функція для прямого проходження через нейронну мережу
// Використовується для обчислення виходу мережі на основі вхідних даних
// Використовує OpenMP секції розкладання для паралельного обчислення
// для прихованого та вихідного шарів
// Використовує динамічне та кероване розкладання для оптимізації продуктивності
void feedforward(double input[INPUT_NODES], 
                double weight1[INPUT_NODES][HIDDEN_NODES],
                double weight2[HIDDEN_NODES][OUTPUT_NODES],
                double bias1[HIDDEN_NODES],
                double bias2[OUTPUT_NODES],
                double hidden[HIDDEN_NODES],
                double output_layer[OUTPUT_NODES]) 
{
    
    // Пряме проходження через прихований шар
    // Використовує динамічне розкладання з кроком 16 для оптимізації продуктивності
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
    
    
    // Пряме проходження через вихідний шар
    // Використовує кероване розкладання для оптимізації продуктивності    
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


// Функція для навчання нейронної мережі
// Використовує зворотне поширення (gradient descent) помилки для оновлення ваг та зсувів
// Використовує OpenMP дерективи для паралельного обчислення
// для обчислення градієнтів та оновлення ваг
// Використовує атомарні операції для безпечного оновлення лічильника правильних відповідей
// Використовує секції для паралельного обчислення градієнтів
// Використовує динамічне та кероване розкладання для оптимізації продуктивності
// Використовує паралельні цикли для оновлення ваг та зсувів
// Використовує секції для паралельного обчислення градієнтів
void train(double input[INPUT_NODES], double output[OUTPUT_NODES], 
    double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], 
    double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], 
    int correct_label, int *correct_counter)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // Пряме проходження через нейронну мережу
    feedforward(input, weight1, weight2, bias1, bias2, hidden, output_layer);

    int index = max_index(output_layer, OUTPUT_NODES);
    
    // Оновлення лічильника правильних відповідей
    // Використовує атомарну операцію для безпечного оновлення
    if (index == correct_label) {
        #pragma omp atomic
        (*correct_counter)++;
    }
    
    
    double delta_output[OUTPUT_NODES];
    
    // Використовує паралельний цикл для обчислення градієнтів входу
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        delta_output[i] = (output[i] - output_layer[i]) * output_layer[i] * (1 - output_layer[i]);
    }
    
    double delta_hidden[HIDDEN_NODES];
    
    // Використовує паралельний цикл для обчислення градієнтів прихованого шару
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
    
    // Оновлення ваг та зсувів
    // Використовує паралельні цикли потоків для оновлення ваг та зсувів
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
    

    // Використовує паралельні цикли потоків для оновлення зсувів прихованого шару
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] += learning_rate * delta_hidden[i];
    }

    // Використовує паралельні цикли потоків для оновлення зсувів виходу
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] += learning_rate * delta_output[i];
    }
    
    // Cекціz для синхронізації потоків
    #pragma omp flush(weight1, weight2, bias1, bias2)
}

// Функція для ініціалізації ваг та зміщень нейронної мережі
// Використовує ініціалізацію Xavier для забезпечення кращої збіжності
// Розподіляє роботу на три секції з використанням OpenMP:
// 1) Ініціалізація ваг між вхідним та прихованим шарами
// 2) Ініціалізація ваг між прихованим та вихідним шарами
// 3) Ініціалізація зміщень для обох шарів
double init_nodes_weight()
{
    
    double xavier_init_hidden = sqrt(6.0 / (INPUT_NODES + HIDDEN_NODES));
    double xavier_init_output = sqrt(6.0 / (HIDDEN_NODES + OUTPUT_NODES));
    
    
    #pragma omp parallel
    {
        // Головний потік розпочинає ініціалізацію
        #pragma omp master
        {
            printf("Weights initialization started by master thread %d\n", omp_get_thread_num());
        }
        
        // Бар'єр синхронізації - всі потоки повинні досягти цієї точки перед продовженням
        #pragma omp barrier
        
        // Розділення роботи на паралельні секції
        #pragma omp sections
        {
            // Секція 1: Ініціалізація ваг між вхідним та прихованим шарами
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
            
            // Секція 2: Ініціалізація ваг між прихованим та вихідним шарами
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
            
            // Секція 3: Ініціалізація зміщень для обох шарів
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
        
        // Один потік виводить повідомлення про завершення ініціалізації
        #pragma omp single
        {
            printf("Weight initialization completed by thread %d\n", 
                   omp_get_thread_num());
        }
    }
    
    return xavier_init_hidden;
}

// Функція для тестування нейронної мережі
// Проводить пряме проходження через мережу та оновлює лічильник правильних відповідей
// Використовується для оцінки точності мережі на тестовому наборі даних
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

// Функція для збереження ваг та зміщень нейронної мережі
// Записує всі ваги та зміщення в бінарний файл для подальшого використання
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

// Функція навчання на основі задач
// Розбиває навчальні дані на підзадачі, які виконуються паралельно
// Використовує OpenMP task для динамічного розподілу роботи між потоками
// що збільшує ефективність використання обчислювальних ресурсів
void task_based_training(int epoch) {
    int correct_train = 0;
    printf("Starting task-based training for epoch %d\n", epoch);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Головний потік створює задачі
            printf("Generating training tasks on thread %d\n", omp_get_thread_num());
            
            // Розбиття на підзадачі по 1000 зображень
            for (int chunk = 0; chunk < NUM_TRAINING_IMAGES; chunk += 1000) {
                #pragma omp task
                {
                    // Окрема підзадача обробляє 1000 зображень
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
                    
                    // Атомарне оновлення загального лічильника правильних відповідей
                    #pragma omp atomic
                    correct_train += local_correct;
                }
            }
            
            // Очікування завершення всіх підзадач
            #pragma omp taskwait
            printf("All training tasks completed for epoch %d\n", epoch);
            printf("Epoch %d : Training Accuracy: %f\n", 
                   epoch, (double)correct_train / NUM_TRAINING_IMAGES);
        }
    }
}


// Функція тестування на основі taskloop
// Використовує дерективу taskloop для динамічного розподілу зображень між потоками
// Розмір зерна (grainsize) 500 вказує, що кожне завдання оброблятиме 500 зображень
// Використовує операцію reduction для безпечного підрахунку правильних прогнозів
void taskloop_based_testing() {
    int correct_predictions = 0;
    printf("Starting taskloop-based testing...\n");
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Створення завдань тестування з розміром зерна 500 зображень
            // і атомарною операцією додавання для безпечного підрахунку
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


// Функція тренування з упорядкованою обробкою партій
// Використовує директиву ordered для послідовного виконання певних операцій у паралельному циклі
// Параметр batch_start визначає початковий індекс партії зображень для обробки
// Параметр batch_size вказує кількість зображень у партії
// Використовує динамічне планування для розподілу роботи між потоками
void ordered_batch_training(int batch_start, int batch_size) {
    printf("Starting ordered batch training from index %d\n", batch_start);
    double batch_time = omp_get_wtime();
    
    
    // Паралельний цикл з директивою ordered для забезпечення певного порядку виконання
    // Використовує динамічне планування для кращого балансування навантаження
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


// Функція оцінки на основі блокувань
// Використовує OpenMP блокування для безпечного оновлення загального лічильника правильних відповідей
// Параметр start_idx визначає початковий індекс зображень для обробки
// Параметр count вказує кількість зображень для обробки
// Кожен потік має свій локальний лічильник, щоб зменшити конкуренцію за блокування
void lock_based_evaluation(int start_idx, int count) {
    omp_lock_t evaluation_lock;
    omp_init_lock(&evaluation_lock);
    
    int local_correct = 0;
    
    #pragma omp parallel
    {
        // Локальний лічильник для кожного потоку
        int thread_correct = 0;
        
        // Директива nowait дозволяє потокам продовжувати роботу, не чекаючи інших
        #pragma omp for nowait
        for (int i = start_idx; i < start_idx + count; i++) {
            if (i < NUM_TEST_IMAGES) {
                int correct_label = max_index(test_labels[i], OUTPUT_NODES);
                test(test_images[i], weight1, weight2, bias1, bias2, 
                     correct_label, &thread_correct);
            }
        }
        
        // Захищене оновлення загального лічильника за допомогою блокування
        omp_set_lock(&evaluation_lock);
        local_correct += thread_correct;
        omp_unset_lock(&evaluation_lock);
        
        // Бар'єр синхронізації - чекаємо, поки всі потоки закінчать обробку
        #pragma omp barrier
    
        // Однопотокова секція для виведення результатів
        #pragma omp single
        {
            printf("Lock-based evaluation completed. Correct: %d/%d\n", 
                   local_correct, count);
        }
    }
    
    // Звільнення ресурсів блокування
    omp_destroy_lock(&evaluation_lock);
}


// Головна функція програми
// Керує всім процесом навчання та тестування нейронної мережі
// Включає: ініціалізацію ваг, завантаження даних MNIST, навчання,
// збереження моделі та тестування на різних підходах
int main()
{
    int correct_predictions = 0;
    
    // Встановлення стабільного зерна для генератора випадкових чисел
    srand(42);     
    
    // Отримання максимальної кількості потоків OpenMP
    int num_threads = omp_get_max_threads();
    printf("Running with %d OpenMP threads\n", num_threads);
    
    // Початок обліку часу виконання програми
    double start_time = omp_get_wtime();
    
    // Ініціалізація ваг та зміщень нейронної мережі
    init_nodes_weight();

    double init_time = omp_get_wtime();
    
    // Завантаження тренувальних та тестових даних MNIST
    load_mnist();
    double load_time = omp_get_wtime();
    printf("MNIST data loaded in %.2f seconds\n", load_time - init_time);

    printf("Starting training...\n");
    
    // Перша половина епох використовує послідовний підхід до навчання
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
    
    
    // Друга половина епох використовує паралельний підхід на основі задач
    for(int epoch=NUMBER_OF_EPOCHS/2; epoch < NUMBER_OF_EPOCHS; epoch++)
    {
        // Використання паралельного тренування на основі задач
        task_based_training(epoch);
        
        // Кожні 10 епох використовуємо також ordered batch тренування
        if (epoch % 10 == 0) {
            ordered_batch_training(0, 5000);
        }
    }
    
    // Збереження ваг та зміщень навченої моделі у файл
    save_weights_biases("model.bin");
    
    // Вимірювання часу навчання
    double train_end_time = omp_get_wtime();
    printf("Training completed in %.2f seconds\n", train_end_time - start_time);

    // Інтерактивне тестування окремих зображень на запит користувача
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

    // Тестування нейронної мережі на всіх тестових зображеннях
    correct_predictions = 0;
    printf("Testing the network on %d images...\n", NUM_TEST_IMAGES);
    
    // Початок обліку часу тестування
    double test_start_time = omp_get_wtime();
    
    // Тестування з використанням різних підходів
    correct_predictions = 0;
    int half_test = NUM_TEST_IMAGES / 2;
    
    // Використання методу з блокуваннями для тестування частини зображень
    lock_based_evaluation(0, 1000);
    
    // Послідовне тестування для першої половини тестових зображень
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
    
    // Паралельне тестування для другої половини тестових зображень
    printf("Starting taskloop-based testing for remaining images...\n");
    taskloop_based_testing();
    
    // Завершення та виведення результатів тестування
    double test_end_time = omp_get_wtime();
    printf("Testing completed in %.2f seconds\n", test_end_time - test_start_time);
    printf("Total execution time: %.2f seconds\n", test_end_time - start_time);
    printf("Correct predictions: %d/%d\n", correct_predictions, NUM_TEST_IMAGES);
    printf("Total test images: %d\n", NUM_TEST_IMAGES);
    
    return 0;
}

