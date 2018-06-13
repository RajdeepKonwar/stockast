# Stockast [![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FRajdeepKonwar%2Fstockast.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FRajdeepKonwar%2Fstockast?ref=badge_shield)
## Stock Market Forecasting using Parallel Monte-Carlo Simulations and Machine Learning

### Requirements
(Steps 3 & 4 required only if you're missing the file "gnuplot-iostream.h")
1. gcc version 6.3.0 or higher  
Local machine (with root privileges; tested on an Ubuntu laptop)
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-6
sudo apt install g++-6
sudo update-alternatives --instal /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ /usr/bin/g++-6
```
Remote machine (without root privileges; tested on [Comet](http://www.sdsc.edu/support/user_guides/comet.html))
```
wget https://ftp.gnu.org/gnu/gcc/gcc-6.3.0/gcc-6.3.0.tar.gz
tar -xvzf gcc-6.3.0.tar.gz
cd gcc-6.3.0
./contrib/download_prerequisites
cd ..
mkdir gcc-build && cd gcc-build
../gcc-6.3.0/configure --disable-multilib -v --prefix=path/to/gcc-6.3.0
make
make install
echo 'export PATH=/(path to gcc-6.3.0)/bin/:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/(path to gcc-6.3.0)/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/(path to gcc-6.3.0)/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```
2. Boost 1.64.0 or higher  
Local machine
```
sudo apt-get update
sudo apt-get install libboost-all-dev
```
Remote machine
```
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
tar -xvzf boost_1_64_0.tar.gz
mkdir boost-build
cd boost_1_64_0
./bootstrap.sh --prefix=/path/to/boost-build
./b2 install
echo 'export LD_LIBRARY_PATH=/(path to boost-build)/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```
##### Important: Re-login after step 2
3. Git
```
sudo apt-get update
sudo apt-get install git
```
4. gnuplot-iostream
```
git clone https://github.com/dstahlke/gnuplot-iostream.git
```
Then copy "gnuplot-iostream.h" to run dir

### Compile Instructions
* Local machine
```
make
```
Type `make clean` to clean object file and executable.
* Remote machine
```
g++ stockast.cpp -o stockast -I "path_to"/boost-build/include -L "path_to"/boost-build/lib -lboost_iostreams -lboost_system -lboost_filesystem -fopenmp
```

### Run Instructions
Set the number of threads to be used for computation,
```
export OMP_NUM_THREADS=number_of_threads
```
For example, `export OMP_NUM_THREADS=8`.
Then run the program (0 for no plot & 1 for plot)
```
./stockast 0
./stockast 1
```

### Requirements to launch machine_learning_rnn.py
Python 2 or 3
numpy
pandas
scikit-learn
TensorFlow - 1.0.0
Scikit-Learn

### General info
* The file "data.csv" contains the original stock-price returns taken from the market.
* The file "ml_data.csv" contains the machine-learning predicted stock-price values for the 3 hours.
* The file "opt.csv" contains the output (most likely outcome) price-vector from our code.
* The script "profiling.sh" runs the parallel code from 1 to maximum number of threads and displays walltimes for each while suppressing any plots (i.e. ./stockast 0 option)
* To use the script, type
```
./profiling.sh "number_of_threads"
```
For example, `./profiling.sh 8`.