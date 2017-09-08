/**
 @author Rajdeep Konwar (rkonwar AT ucsd.edu)

 @section DESCRIPTION
 Stock Market Forecasting using parallel Monte-Carlo simulations
 (src:wiki) The Black–Scholes model assumes that the market consists of at least one risky asset,
 usually called the stock, and one riskless asset, usually called the money market, cash, or bond.
 The rate of return on the riskless asset is constant and thus called the risk-free interest rate.

 @section REQUIREMENTS (steps 3 & 4 only if missing file "gnuplot-iostream.h")
 1. gcc version 6.3.0 or higher
  a. UBUNTU: (Needs root privileges)
      <sudo add-apt-repository ppa:ubuntu-toolchain-r/test> <sudo apt update>
      <sudo apt install gcc-6> <sudo apt install g++-6>
      <sudo update-alternatives --instal /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ /usr/bin/g++-6>
  b. OTHERS: (Without root privileges. Add the last 3 instruction lines to end of ~/.bashrc)
      <wget https://ftp.gnu.org/gnu/gcc/gcc-6.3.0/gcc-6.3.0.tar.gz>
      <tar -xvzf gcc-6.3.0.tar.gz> <cd gcc-6.3.0>
      <./contrib/download_prerequisites> <cd ..>
      <mkdir gcc-build && cd gcc-build>
      <../gcc-6.3.0/configure --disable-multilib -v --prefix=path/to/gcc-6.3.0>
      <make> <make install> <export PATH=~/gcc-6.3.0/bin/:$PATH>
      <export LD_LIBRARY_PATH=path/to/gcc-6.3.0/lib:$LD_LIBRARY_PATH>
      <export LD_LIBRARY_PATH=path/to/gcc-6.3.0/lib64:$LD_LIBRARY_PATH>
 2. Boost 1.64.0 or higher
  a. UBUNTU: (needs root privilieges)
      <sudo apt-get update> <sudo apt-get install libboost-all-dev>
  b. OTHERS: (without root privileges. Add the last instruction line to end of ~/.bashrc)
      <wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz>
      <tar -xvzf boost_1_64_0.tar.gz> <mkdir boost-build> <cd boost_1_64_0>
      <./bootstrap.sh --prefix=/path/to/boost-build> <./b2 install>
      <export LD_LIBRARY_PATH=/path/to/boost-build/lib:$LD_LIBRARY_PATH>
 3. Git   <sudo apt-get update> <sudo apt-get install git>
 4. <git clone https://github.com/dstahlke/gnuplot-iostream.git> then copy "gnuplot-iostream.h" to run dir

 @section COMPILE INSTRUCTION
 (Local): g++ stockast.cpp -o stockast -lboost_iostreams -lboost_system -lboost_filesystem -fopenmp
 (comet): g++ stockast.cpp -o stockast -I /path/to/boost-build/include -L /path/to/boost-build/lib -lboost_iostreams -lboost_system -lboost_filesystem -fopenmp

 @section RUN INSTRUCTION
 export OMP_NUM_THREADS=<number of threads>
 ./stockast <plotFlag: 0 for noplot/1 for plot>

 @section LICENSE
 Copyright (c) 2017, Regents of the University of California
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! Header files
#include <iostream>
#include <chrono>
#include <omp.h>
#include <random>
#include <string.h>
#include "gnuplot-iostream.h"

//! Function Declarations
float   calcVolatility( float spot, int timesteps );
float*  find2DMean( float **stockMat, int M, int N );
float   randGen( float mean, float sd );
float*  runBlackScholesModel( float sp, int n, float r, float sig );

//! Main function
int main( int i_argc, char** i_argv ) {
  int     plotFlag;     //! Flag indicating whether to plot or not

  //! Check for input terminal arguments. If none specified, plot by default
  if( i_argc != 2 )
    plotFlag  = 1;
  else
    plotFlag  = atoi( i_argv[1] );

  //! Start time
  clock_t t   = clock();

  //! Variable declaration
  int     i, j;         //! Loop iterators
  int     inLoops;      //! Inner loop iterations
  int     outLoops;     //! Outer loop iterations
  int     timesteps;    //! Stock market time-intervals (min)
  int     numThreads;   //! Number of threads
  float   riskRate;     //! Risk free interest rate (%)
  float   sigma;        //! Market volatility (calculated from data.csv)
  float   spot;         //! Spot price (at t = 0)
  float** stock;        //! Matrix for stock-price vectors per iteration
  float** avgStock;     //! Matrix for mean of stock-price vectors per iteration
  float*  optStock;     //! Vector for most likely outcome stock price
  FILE*   fp;           //! File pointer

  //! Initialization
  timesteps = 180;
  inLoops   = 100;
  outLoops  = 10000;
  spot      = 100.0;
  riskRate  = 0.001;
  sigma     = calcVolatility( spot, timesteps );

  //! Memory allocation
  stock     = new float* [inLoops];
  for( i = 0; i < inLoops; i++ )
    stock[i] = new float [timesteps];

  avgStock  = new float* [outLoops];
  for( i = 0; i < outLoops; i++ )
    avgStock[i] = new float [timesteps];

  optStock  = new float [timesteps];

  //! Welcome message
  std::cout << "--Welcome to Stockast: Stock Forecasting Tool--\n";
  std::cout << "  Parth Shah, Premanand Kumar, Rajdeep Konwar  \n\n";
  std::cout << "  Using market volatility = " << sigma << std::endl;

  //! Parallel region with each thread having its own instance of variable 'i',
#pragma omp parallel private(i)
  {
    //! Only one thread (irrespective of thread id) handles this region
#pragma omp single
    {
      numThreads  = omp_get_num_threads();
      std::cout << "  Using " << numThreads << " thread(s)..\n";
      omp_set_num_threads( numThreads );
    }

    /**
     Parallel for loop with dynamic scheduling, i.e. each thread
     grabs "chunk" iterations until all iterations are done.
     Faster threads are assigned more iterations (not Round Robin)
     */
#pragma omp for schedule(dynamic)
    for( i = 0; i < outLoops; i++ ) {

      /**
       Using Black Scholes model to get stock price every iteration
       Returns data as a column vector having rows=timesteps
       */
      for( j = 0; j < inLoops; j++ )
        stock[j]  = runBlackScholesModel( spot, timesteps, riskRate, sigma );

      //! Stores average of all estimated stock-price arrays
      avgStock[i] = find2DMean( stock, inLoops, timesteps );
    }
    //! --> Implicit omp barrier <--
  }

  //! Average of all the average arrays
  optStock  = find2DMean( avgStock, outLoops, timesteps );

  //! Write optimal outcome to disk
  fp  = std::fopen( "opt.csv", "w" );
  for( i = 0; i < timesteps; i++ )
    fprintf( fp, "%f\n", optStock[i] );

  //! Close the file pointer
  fclose( fp );

  //! Memory deallocation
  for( i = 0; i < inLoops; i++ )
    delete[] stock[i];
  delete[] stock;

  for( i = 0; i < outLoops; i++ )
    delete[] avgStock[i];
  delete[] avgStock;

  delete[] optStock;

  //! Plot the most likely outcome in Gnuplot if plotFlag = 1
  if( plotFlag ) {
    Gnuplot gp( "gnuplot -persist" );
    gp << "set grid\n";
    gp << "set title 'Stock Market Forecast (3 hrs)'\n";
    gp << "set xlabel 'Time (min)'\n";
    gp << "set ylabel 'Price'\n";
    gp << "plot 'opt.csv' w l title 'Predicted Stock Price'\n";
  }

    //! Finish time
  t = clock() - t;
  std::cout << "  Time taken = " << (float)t / CLOCKS_PER_SEC << "s\n\n";

  return 0;
}

//! Simulates Black Scholes model
float* runBlackScholesModel( float sp, int n, float r, float sig ) {
  int   i;                          //! Loop counter
  float mean    = 0.0, sd = 1.0;    //! Mean and standard deviation
  float deltaT  = 1.0 / n;          //! Timestep
  float *z      = new float [n-1];  //! Array of normally distributed random nos.
  float *st     = new float [n];    //! Array of stock price at diff. times
  st[0]         = sp;               //! Stock price at t=0 is spot price

  //! Populate array with random nos.
  for( i = 0; i < n - 1; i++ )
    z[i]  = randGen( mean, sd );

  //! Apply Black Scholes equation to calculate stock price at next timestep
  for( i = 0; i < n - 1; i++ )
    st[i+1] = st[i] * exp( ((r - (pow( sig, 2.0 ) / 2)) * deltaT) + (sig * z[i] * sqrt( deltaT )) );

  delete[] z;

  return st;
}

/**
 Finds mean of a 2D array across first index (inLoops)
 M is in/outLoops and N is timesteps
 */
float* find2DMean( float **matrix, int M, int N ) {
  int     i, j;
  float*  avg = new float [N];
  float   sum = 0.0;

  for( i = 0; i < N; i++ ) {

    /**
     A private copy of 'sum' variable is created for each thread.
     At the end of the reduction, the reduction variable is applied to
     all private copies of the shared variable, and the final result
     is written to the global shared variable.
     */
#pragma omp parallel for private(j) reduction(+:sum)
    for( j = 0; j < M; j++ ) {
      sum += matrix[j][i];
    }

    //! Calculating average across columns
    avg[i] = sum / M;
    sum = 0.0;
  }

  return avg;
}

/**
 Generates a random number seeded by system clock based on standard
 normal distribution on taking mean 0.0 and standard deviation 1.0
 */
float randGen( float mean, float sd ) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator( seed );
  std::normal_distribution< float > distribution( mean, sd );

  return distribution( generator );
}

//! Calculates volatility from ml_data.csv file
float calcVolatility( float spot, int timesteps ) {
  int   i, len = timesteps - 1;
  char  line[4096], *token;
  float mean, sum, sd, priceArr[len];
  FILE* fp;

  //! Open ml_data.csv in read-mode, exit on fail
  fp = std::fopen( "ml_data.csv", "r" );
  if( !fp ) {
    std::cout << "Cannot open ml_data.csv!\n";
    exit( EXIT_FAILURE );
  }

  //! Read the first line then close file
  if( fgets( line, sizeof( line ), fp ) == NULL ) {
    std::cout << "Cannot read from ml_data.csv!\n";
    fclose( fp );
    exit( EXIT_FAILURE );
  }
  fclose( fp );

  //! Get the return values of stock from file (min 2 to 180)
  for( i = 0; i < len; i++ ) {
    if( i == 0 )
      token   = strtok( line, "," );
    else
      token   = strtok( NULL, "," );
    priceArr[i] = atof( token );
  }

  sum = spot;
  //! Find mean of the estimated minute-end prices
  for( i = 0; i < len; i++ )
    sum += priceArr[i];
  mean = sum / (len + 1);

  //! Calculate market volatility as standard deviation
  sum = pow( (spot - mean), 2.0 );
  for( i = 0; i < len; i++ )
    sum += pow( (priceArr[i] - mean), 2.0 );
  sd = sqrt( sum );

  //! Return as percentage
  return (sd / 100.0);
}
