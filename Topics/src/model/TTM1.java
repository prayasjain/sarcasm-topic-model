package model;

import utils.TopicModelUtils;


public class TTM1 {
	
	public int Z;  // number of topics
	public int W;  // vocabulary
	public int U;  // number of users
	public int D;  // number of documents (clicked urls)
	public long N; // total word occurrences
	
	public double au;  // topic smoothing hyperparameter for users
	public double ad;  // topic smoothing hyperparameter for documents 
	public double b;   // term smoothing hyperparamter \beta 
	
	public int[][][] w_uij; // w_uij[u][i][j] = j'th word in the i'th query by the u'th user
	public int[][][] z_uij; // z_uij[u][i][j] = topic assignment to j'th position in the i'th query by the u'th user
	public int[][] d_ui;    // d_ui[u][i] = document for i'th query by the u'th user
	
	public int[][] N_zu; // N_zu[z][u] = count of z'th topic for the u'th user
	public int[][] N_zd; // N_zd[z][d] = count of z'th topic for the d'th document
	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	public int[] N_u;    // N_u[u] = count of u'th user
	public int[] N_d;    // N_d[d] = count of d'th document
	public int[] N_z;    // N_z[z] = count of z'th topic
		
	// private variables
	private double b_on_W;
	private double au_on_Z;
	private double ad_on_Z;
	
	
	public void estimate(int[][][] w_uij, int[][] d_ui, int W, int D, long N, int Z, double au, double ad, double b, int iterations, int step) {
		
		this.w_uij = w_uij;
		this.d_ui = d_ui;
		this.W = W;
		this.U = w_uij.length;
		this.D = D;
		this.N = N;
		this.Z = Z;
		this.au = au;
		this.ad = ad;
		this.b = b;

		b_on_W = b/W;
		au_on_Z = au/Z;
		ad_on_Z = ad/Z;

		// initialize latent variable assignment and count matrices
		z_uij = new int[U][][];
		N_zu = new int[Z][U];
		N_zd = new int[Z][D];
		N_wz = new int[W][Z];
		N_u = new int[U];
		N_d = new int[D];
		N_z = new int[Z];
		for (int u=0; u<U; u++) {  // user u
			z_uij[u] = new int[w_uij[u].length][];
			for (int i=0; i<w_uij[u].length; i++) { // query i
				z_uij[u][i] = new int[w_uij[u][i].length];
				int d = d_ui[u][i];
				for (int j=0; j<w_uij[u][i].length; j++) { // position j
					int z = (int) (Z * Math.random());
					z_uij[u][i][j] = z;
					N_zu[z][u]++;
					N_zd[z][d]++;
					N_wz[w_uij[u][i][j]][z]++;
					N_u[u]++;
					N_d[d]++;
					N_z[z]++;
				}	
			}
		}
		
		// perform Gibbs sampling
		for (int iteration=0; iteration<iterations; iteration++) {
			for (int u=0; u<U; u++) {  // user u
				for (int i=0; i<w_uij[u].length; i++) { // query i
					int d = d_ui[u][i];
					for (int j=0; j<w_uij[u][i].length; j++) { // position j
						int w = w_uij[u][i][j];
						int z = z_uij[u][i][j];
						
						// remove last value  
						N_zu[z][u]--;
						N_zd[z][d]--;
						N_wz[w][z]--;
						N_z[z]--;
						// calculate distribution
						double[] p = new double[Z];
						double total = 0;
						for (int k=0; k<Z; k++) {
							p[k] = ( (N_wz[w][k] + b_on_W)/(N_z[k] + b) ) * (N_zu[k][u] + au_on_Z) * (N_zd[k][d] + ad_on_Z) / N_z[k];
							total += p[k];
						}
						// resample 
						double val = total * Math.random();
						z = 0; while ((val -= p[z]) > 0) z++;
						// update latent variable and counts
						z_uij[u][i][j] = z;
						N_zu[z][u]++;
						N_zd[z][d]++;
						N_wz[w][z]++;
						N_z[z]++;
					}
				}	
			}	
			if (iteration%step==0) System.out.println("TTM1: iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.saveMatrix(estimateP_w_z(),"P_w_z.data");
		TopicModelUtils.saveMatrix(estimateP_z_d(),"P_z_d.data");
		TopicModelUtils.saveMatrix(estimateP_z_u(),"P_z_u.data");
		TopicModelUtils.saveVector(estimateP_z(),"P_z.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");
		
	}	
		
	
	
	public double logLikelihood() {
		double ll = 0.0;
		for (int u=0; u<U; u++) {  // user u
			for (int i=0; i<w_uij[u].length; i++) { // query i
				int d = d_ui[u][i];
				// estimate probability distribution p(z|u,d) 
				double[] p_z = new double[Z];
				double total = 0;
				for (int k=0; k<Z; k++) {
					p_z[k] = (N_zu[k][u] + au_on_Z) * (N_zd[k][d] + ad_on_Z) / N_z[k];
					total += p_z[k];
				}
				// normalize distribution and take log
				for (int k=0; k<Z; k++) p_z[k] = Math.log(p_z[k]/total);
				// use distribution to calculate likelihood
				for (int j=0; j<w_uij[u][i].length; j++) { // position j
					int z = z_uij[u][i][j];
					int w = w_uij[u][i][j];
					ll += Math.log( (N_wz[w][z] + b_on_W)/(N_z[z] + b) );
					ll += p_z[z];
				}
			}
		}
		return ll;
	}
	
	
	public double[][] estimateP_w_z() {
		double[][] p_w_z = new double[W][Z];
		for (int w=0; w<W; w++) 
			for (int z=0; z<Z; z++) 
				p_w_z[w][z] = (N_wz[w][z] + b_on_W)/(N_z[z] + b);
		return p_w_z;
	}
	
	
	public double[][] estimateP_z_d() {
		double[][] p_z_d = new double[Z][D];
		for (int d=0; d<D; d++)
			for (int z=0; z<Z; z++)
				p_z_d[z][d] = (N_zd[z][d] + ad_on_Z)/(N_d[d] + ad);
		return p_z_d;
	}
	
	
	public double[][] estimateP_z_u() {
		double[][] p_z_u = new double[Z][U];
		for (int u=0; u<U; u++)
			for (int z=0; z<Z; z++)
				p_z_u[z][u] = (N_zu[z][u] + au_on_Z)/(N_u[u] + au);
		return p_z_u;
	}
	
	
	public double[] estimateP_z() {
		double[] p_z = new double[Z];
		for (int z=0; z<Z; z++)
			p_z[z] = N_z[z]/(double) N;
		return p_z;
	}
	
	
	public double[] estimateP_d() {
		double[] p_d = new double[D];
		for (int d=0; d<D; d++)
			p_d[d] = N_d[d]/(double) N;
		return p_d;
	}


}
