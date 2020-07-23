package model;

import utils.TopicModelUtils;


// A Segmentation Topic Model
public class STM {
	
	public int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public long N; // total word occurrences
	
	public double a; // topic smoothing hyperparameter \alpha
	public double b; // term smoothing hyperparamter \beta 
	
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document
	
	public int[][] N_zd; // N_zd[z][d] = count of z'th topic in d'th document
	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	public int[] N_z;    // N_z[z] = count of z'th topic
	public int[] N_d;    // N_d[d] = length of document d
	 
	public double[][] P_w_z;
	public double[][] P_z_d;
		
	public int num_samples;
	
	// private variables
	private double b_on_W;
	private double a_on_Z;

	
	public double[][] estimate(int[][] w_di, int W, int D, long N, int Z, double a, double b, int burnIn, int samples, int step) {
		
		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		this.a = a;
		this.b = b;
		num_samples = 0;
		b_on_W = b/W;
		a_on_Z = a/Z;
		
		
		// initialize latent variable assignment and count matrices
		z_di = new int[D][];
		N_zd = new int[Z][D];
		N_wz = new int[W][Z];
		N_z = new int[Z];
		N_d = new int[D];
		
		P_w_z = new double[W][Z];
		P_z_d = new double[Z][D];
		
		for (int d=0; d<D; d++) {  // document d
			int I = w_di[d].length;
			z_di[d] = new int[I];
			N_d[d] = I;
			for (int i=0; i<I; i++) { // position i
				int z = (int) (Z * Math.random());
				z_di[d][i] = z;
				N_zd[z][d]++;
				N_wz[w_di[d][i]][z]++;
				N_z[z]++;
			}
		}
		
		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {
			for (int d=0; d<D; d++) { // document d
				for (int i=0; i<w_di[d].length; i++) { // position i
					
					int w = w_di[d][i];
					int z = z_di[d][i];
					
					// remove last value  
					N_zd[z][d]--;
					N_wz[w][z]--;
					N_z[z]--;
					
					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					double[] p = new double[Z];
					double total = 0;
					for (z=0; z<Z; z++) {
						p[z] = ( (N_wz[w][z] + b_on_W)/(N_z[z] + b) ) * (N_zd[z][d] + a_on_Z);
						total += p[z];
					}
					
					// resample 
					double val = total * Math.random();
					z = 0; while ((val -= p[z]) > 0) z++;
					
					// update latent variable and counts
					z_di[d][i] = z;
					
					N_zd[z][d]++;
					N_wz[w][z]++;
					N_z[z]++;
					
				}	
			}	

			// update parameter estimates
			if (iteration >= burnIn) {
				for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] += (N_wz[w][z] + b_on_W)/(N_z[z] + b);
				for (int d=0; d<D; d++) for (int z=0; z<Z; z++) P_z_d[z][d] += (N_zd[z][d] + a_on_Z)/(N_d[d] + a);
			}
			
			if (iteration%step==0) System.out.println("iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}
		
		// normalize parameter estimates
		for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_z_d[z][d] /= samples;
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.saveMatrix(P_w_z,"P_w_z.data");
		TopicModelUtils.saveMatrix(P_z_d,"P_z_d.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");
		
		return P_w_z;		
	}	
		
	
	public double logLikelihood() {
		double ll = 0;
		for (int d=0; d<D; d++) { // document d
			for (int i=0; i<N_d[d]; i++) { // position i
				int z = z_di[d][i];
				int w = w_di[d][i];
				ll += Math.log( (N_wz[w][z] + b_on_W)/(N_z[z] + b) ); 
				ll += Math.log( (N_zd[z][d] + a_on_Z)/(N_d[d] + a) );
			}
		}
		return ll;
	}
	

	public double[] estimateP_d(){
		return estimateP_d(false);
	}
	
	public double[] estimateP_d(boolean smooth) { // Do some Jelinek-Mercer smoothing
		double lambda = 0.5;
		double uniform = 1/D;
		double[] p_d = new double[D];
		for (int d=0; d<D; d++)
			p_d[d] = smooth ? (lambda*((double)N_d[d]/N))+((1-lambda)*uniform) : (double)N_d[d]/N;
		return p_d;
	}
	
	
}
