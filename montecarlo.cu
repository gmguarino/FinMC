#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <exception>

#include <time.h>
#include <math.h>
#include <curand.h>
#include <cuda_runtime.h>
#include "cuda_array.hh"
#include "mc_kernel.hh"



int arg_parse(int argc, char* argv[], char* argument)
{
    for (size_t i = 0; i < argc; i++)
    {
        if (strcmp(argument, argv[i]))
        {
            return 1;
        }
    }
    return 0;
}


int main(int argc, char* argv[])
{
    try
    {
        
        bool plot = false;
        FILE  *myfile;

        
        if (arg_parse(argc, argv, (char *) "plot"))
        {
            plot = true;
            myfile = fopen("data.dat", "w+");
        }
        
        
        /*Number of parallel simulations run by the GPU*/
        const size_t NPATHS = 1000000;

        /* Steps for the simulation (one per day of the year)  */
        const size_t NSTEPS = 365;

        /* N_NORMALS is the number of normally distributed random
        numbers necessary for the simulation */
        const size_t N_NORMALS = NPATHS * NSTEPS;


        /* Now we can define some market parameters*/

        const float T = 1.0f; // Time to maturity... one year
        const float B = 95.0f; // Barrier level
        const float K = 100.0f; // Strike price
        const float S0 = 100.0f; // Initial stock price
        const float sigma = 0.2f; // volatility
        const float mu = 0.1f; // Yearly expected return
        const float r = 0.05f; // risk free interest rate

        /* Simulation parameters */

        float dt =  T / (float)NSTEPS; // time step size
        float sqrtdt = sqrt(dt);

        /* Data Structures */
        float *stock_prices; // CPU array where to retrieve final results
        stock_prices = (float *)malloc(NPATHS * sizeof(float));
        CudaArray<float> device_stock_prices(NPATHS); // GPU array to calculate stock prices in parallel
        CudaArray<float> device_normal_distributed(N_NORMALS); // GPU array with rand numbers for mc sim

        /* Declare a random number generator for the GPU */
        curandGenerator_t curandGenerator;
        /* Set the generator */
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
        /* Seed the generator */
        curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

        /* Generate normally distributed random number array of size N_NORMALS
            with mean 0 and stddev = sqrt(dt) */
        curandGenerateNormal(
            curandGenerator,
            device_normal_distributed.getData(), 
            N_NORMALS, 
            0.0f,
            sqrtdt);


        double t = double(clock())/CLOCKS_PER_SEC;
        
        /* Blocks of 32x32 threads */
        dim3 BLOCK_SHAPE(32, 32);
        dim3 GRID_SHAPE(
            ceil(sqrt(NPATHS) / BLOCK_SHAPE.x),
            ceil(sqrt(NPATHS) / BLOCK_SHAPE.y)
        );

        /* Calling the Kernel wrapper function */
        mc_call(
            device_stock_prices.getData(),
            T,
            K,
            B,
            S0,
            sigma,
            mu,
            r,
            dt,
            device_normal_distributed.getData(),
            NSTEPS,
            NPATHS,
            BLOCK_SHAPE,
            GRID_SHAPE
        );

        /* Wait for all threads to finish */
        cudaDeviceSynchronize();

        double time_taken = double(clock())/CLOCKS_PER_SEC - t;

        /* Copy data to CPU */
        device_stock_prices.get(stock_prices, NPATHS);

        float price = 0;

        /* Save data to file */
        for (size_t i = 0; i < NPATHS; i++)
        {
            price += stock_prices[i];
            if (plot)
            {
                fprintf(myfile, "%.4f, ", stock_prices[i]);
            }
        }

        /* Calculate average price simulated */
        price /= NPATHS;
        
        std::cout << "\n";
        std::cout << "--------------------Results--------------------\n";
        std::cout << "Number of simulations: " << NPATHS << "\n";
        std::cout << "Initial Price: " << S0 << "\n";
        std::cout << "Strike: " << K << "\n";
        std::cout << "Barrier: " << B << "\n";
        std::cout << "Maturity time: " << T << " years\n";
        std::cout << "Risk-free interest rate: " << r << "\n";
        std::cout << "Annual drift: " << mu << "\n";
        std::cout << "Volatility: " << sigma << "\n";
        std::cout << "\n";
        std::cout << "Calculated price: " << price << "\n";
        std::cout << "Computation Time on GPU: " << time_taken * 1e3 << " ms\n";
        std::cout << "\n";

        /* Clean up */
        curandDestroyGenerator( curandGenerator ) ;
        free(stock_prices);
        if (plot)
        {
            fclose(myfile);
        }

    }
    catch(const std::exception& e)
    {
        /* Print out any error */
        std::cerr << e.what() << '\n';
    }

    return 0;
}
