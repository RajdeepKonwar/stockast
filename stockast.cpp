/**
* @file This file is part of stockast.
*
* @section LICENSE
* MIT License
*
* Copyright (c) 2017-2022 Rajdeep Konwar
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* @section DESCRIPTION
* Stock Market Forecasting using parallel Monte-Carlo simulations
* (src:wikipedia) The Black–Scholes model assumes that the market consists of
* at least one risky asset, usually called the stock, and one riskless asset,
* usually called the money market, cash, or bond. The rate of return on the
* riskless asset is constant and thus called the risk-free interest rate.
**/

// Header files
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>

//----------------------------------------------------------------------------
// Calculates volatility from data.csv file
//----------------------------------------------------------------------------
float calculate_volatility(float spot_price, int time_steps)
{
    // Open data.csv in read-mode, exit on fail
    const char* file_name = "data.csv";
    std::ifstream file_ptr;
    file_ptr.open(file_name, std::ifstream::in);
    if (!file_ptr.is_open())
    {
        std::cerr << "Cannot open data.csv! Exiting..\n";
        exit(EXIT_FAILURE);
    }

    std::string line;
    // Read the first line then close file
    if (!std::getline(file_ptr, line))
    {
        std::cerr << "Cannot read from data.csv! Exiting..\n";
        file_ptr.close();
        exit(EXIT_FAILURE);
    }
    file_ptr.close();

    int i = 0, len = time_steps - 1;
    std::unique_ptr<float[]> priceArr = std::make_unique<float[]>(time_steps - 1);
    std::istringstream iss(line);
    std::string token;

    // Get the return values of stock from file (min 2 to 180)
    while (std::getline(iss, token, ','))
        priceArr[i++] = std::stof(token);

    float sum = spot_price;
    // Find mean of the estimated minute-end prices
    for (i = 0; i < len; i++)
        sum += priceArr[i];
    float mean_price = sum / (len + 1);

    // Calculate market volatility as standard deviation
    sum = powf((spot_price - mean_price), 2.0f);
    for (i = 0; i < len; i++)
        sum += powf((priceArr[i] - mean_price), 2.0f);

    float std_dev = sqrtf(sum);

    // Return as percentage
    return std_dev / 100.0f;
}

/** ---------------------------------------------------------------------------
    Finds mean of a 2D array across first index (in_loops)
    M is in/out_loops and N is time_steps
----------------------------------------------------------------------------*/
float* find_2d_mean(float** matrix, int num_loops, int time_steps)
{
    int j;
    float* avg = new float[time_steps];
    float sum = 0.0f;

    for (int i = 0; i < time_steps; i++)
    {
        /** A private copy of 'sum' variable is created for each thread.
            At the end of the reduction, the reduction variable is applied to
            all private copies of the shared variable, and the final result
            is written to the global shared variable. **/
#pragma omp parallel for private(j) reduction(+:sum)
        for (j = 0; j < num_loops; j++)
        {
            sum += matrix[j][i];
        }

        // Calculating average across columns
        avg[i] = sum / num_loops;
        sum = 0.0f;
    }

    return avg;
}

/** ---------------------------------------------------------------------------
    Generates a random number seeded by system clock based on standard
    normal distribution on taking mean 0.0 and standard deviation 1.0
----------------------------------------------------------------------------*/
float rand_gen(float mean, float std_dev)
{
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(static_cast<unsigned int>(seed));
    std::normal_distribution<float> distribution(mean, std_dev);
    return distribution(generator);
}

//----------------------------------------------------------------------------
// Simulates Black Scholes model
//----------------------------------------------------------------------------
float* run_black_scholes_model(float spot_price, int time_steps, float risk_rate, float volatility)
{
    float  mean = 0.0f, std_dev = 1.0f;                                             // Mean and standard deviation
    float  deltaT = 1.0f / time_steps;                                              // Timestep
    std::unique_ptr<float[]> norm_rand = std::make_unique<float[]>(time_steps - 1); // Array of normally distributed random nos.
    float* stock_price = new float[time_steps];                                     // Array of stock price at diff. times
    stock_price[0] = spot_price;                                                    // Stock price at t=0 is spot price

    // Populate array with random nos.
    for (int i = 0; i < time_steps - 1; i++)
        norm_rand[i] = rand_gen(mean, std_dev);

    // Apply Black Scholes equation to calculate stock price at next timestep
    for (int i = 0; i < time_steps - 1; i++)
        stock_price[i + 1] = stock_price[i] * exp(((risk_rate - (powf(volatility, 2.0f) / 2.0f)) * deltaT) + (volatility * norm_rand[i] * sqrtf(deltaT)));

    return stock_price;
}

//----------------------------------------------------------------------------
// Main function
//----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    clock_t t = clock();

    int in_loops = 100;     // Inner loop iterations
    int out_loops = 10000;  // Outer loop iterations
    int time_steps = 180;   // Stock market time-intervals (min)

    // Matrix for stock-price vectors per iteration
    float** stock = new float* [in_loops];
    for (int i = 0; i < in_loops; i++)
        stock[i] = new float[time_steps];

    // Matrix for mean of stock-price vectors per iteration
    float** avg_stock = new float* [out_loops];
    for (int i = 0; i < out_loops; i++)
        avg_stock[i] = new float[time_steps];

    // Vector for most likely outcome stock price
    float* opt_stock = new float[time_steps];

    float risk_rate = 0.001f;   // Risk free interest rate (%)
    float spot_price = 100.0f;  // Spot price (at t = 0)

    // Market volatility (calculated from data.csv)
    float volatility = calculate_volatility(spot_price, time_steps);

    // Welcome message
    std::cout << "--Welcome to Stockast: Stock Forecasting Tool--\n";
    std::cout << "  Copyright (c) 2017-2022 Rajdeep Konwar\n\n";
    std::cout << "  Using market volatility = " << volatility << std::endl;

    int i;
    // Parallel region with each thread having its own instance of variable 'i',
#pragma omp parallel private(i)
    {
        // Only one thread (irrespective of thread id) handles this region
#pragma omp single
        {
            int numThreads = omp_get_num_threads(); // Number of threads
            std::cout << "  Using " << numThreads << " thread(s)\n\n";
            std::cout << "  Have patience! Computing..";
            omp_set_num_threads(numThreads);
        }

        /** Parallel for loop with dynamic scheduling, i.e. each thread
            grabs "chunk" iterations until all iterations are done.
            Faster threads are assigned more iterations (not Round Robin) **/
#pragma omp for schedule(dynamic)
        for (i = 0; i < out_loops; i++)
        {
            /** Using Black Scholes model to get stock price every iteration
                Returns data as a column vector having rows=time_steps  **/
            for (int j = 0; j < in_loops; j++)
                stock[j] = run_black_scholes_model(spot_price, time_steps, risk_rate, volatility);

            // Stores average of all estimated stock-price arrays
            avg_stock[i] = find_2d_mean(stock, in_loops, time_steps);
        }
        //---> Implicit omp barrier <--
    }

    // Average of all the average arrays
    opt_stock = find_2d_mean(avg_stock, out_loops, time_steps);

    // Write optimal outcome to disk
    std::ofstream file_ptr;
    file_ptr.open("opt.csv", std::ofstream::out);
    if (!file_ptr.is_open())
    {
        std::cerr << "Couldn't open opt.csv! Exiting..\n";
        return EXIT_FAILURE;
    }

    for (i = 0; i < time_steps; i++)
        file_ptr << opt_stock[i] << "\n";
    file_ptr.close();

    for (i = 0; i < in_loops; i++)
        delete[] stock[i];
    delete[] stock;

    for (i = 0; i < out_loops; i++)
        delete[] avg_stock[i];
    delete[] avg_stock;

    delete[] opt_stock;

    t = clock() - t;
    std::cout << " done!\n  Time taken = " << static_cast<float>(t / CLOCKS_PER_SEC) << "s";

    return getchar();
}
