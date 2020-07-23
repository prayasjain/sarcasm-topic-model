package utils;

import java.util.ArrayList;
import java.util.List;

import be.ac.ulg.montefiore.run.distributions.MultiGaussianDistribution;

public class LogisticNormal {

	public MultiGaussianDistribution mg;
	
	public LogisticNormal()
	{
		
	}
	
	public LogisticNormal(double[] d_mean, double[][] d_covar)
	{
		int Z = d_mean.length;
		mg = new MultiGaussianDistribution(d_mean, d_covar);
			
	}
	
	public double[] sample()
	{
		double[] val = mg.generate();
		double[] new_val = new double[val.length + 1];
		
	//	this.printArray(val);
		double denominator = 1.0d;
		
		for (int i = 0; i <val.length; i++)
		{
			denominator += Math.exp(val[i]);
			
		}
		
		new_val[0] = 1/denominator;
		
		for (int i = 1; i<= val.length; i++)
		{
			new_val[i] = Math.exp(val[i-1])/denominator;
		}
		
		
		return new_val;
	}
	
	public void printArray(double[] arr)
	{
		System.out.print("[ ");
		for (int i = 0; i < arr.length; i++)
		{
			System.out.print(arr[i]+" ");
		}
		System.out.println("]");
	}
	
	public double[] averagedsample(double[] d_mean, double[][] d_covar, int Z)
	{
		int num_samples = 20;
		LogisticNormal ln = new LogisticNormal(d_mean, d_covar);
		
		double[] final_sample = new double[Z];
		
		for (int i = 0; i < num_samples; i++)
		{
			double[] curr_sample = ln.sample();
			
			for (int z = 0; z < Z; z++)
				final_sample[z] += curr_sample[z];
		}
		
		for (int z=0;z<Z;z++)
			final_sample[z] /= num_samples;
		
		return final_sample;
	
	}
	
	public static void main(String[] args)
	{
		double[] d_mean = {1,1,1};
		double[][] d_covar = {{20,10,0},{10,20,0},{0,0,20}};
		
		LogisticNormal ln = new LogisticNormal(d_mean, d_covar);
		
		int num_samples = 20;
		
		for (int i = 0; i < num_samples; i++)
		{
			double[] curr_sample = ln.sample();
			
			ln.printArray(curr_sample);
			
		}
	}
}
