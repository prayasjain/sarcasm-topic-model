package model;

import utils.TopicModelUtils;
import java.io.*;

public class PTM3_sequential {
	
	public int Y; // number of user interests
	public int Z; // number of document topics
	public int W; // vocabulary
	public int U; // number of users
	public int D; // number of documents (clicked urls)
	public long N; // total word occurrences
	
	public double au; // interest smoothing hyperparameter for users
	public double ad; // topic smoothing hyperparameter for documents 
	public double b; // term smoothing hyperparamter \beta 
	
	public int[][][] w_uij; // w_uij[u][i][j] = j'th word in the i'th query by the u'th user
	public int[][][] y_uij; // y_uij[u][i][j] = interest assignment to j'th position in the i'th query by the u'th user
	public int[][][] z_uij; // z_uij[u][i][j] = topic assignment to j'th position in the i'th query by the u'th user
	public int[][] d_ui;    // d_ui[u][i] = document for i'th query by the u'th user
	
	public int[][] N_yu;    // N_yu[y][u] = count of y'th interest for the u'th user
	public int[][] N_zd;    // N_zd[z][d] = count of z'th topic for the d'th document
	public int[][][] N_wyz; // N_wyz[w][y][z] = count of w'th word for y'th interest and z'th topic
	public int[] N_u;       // N_u[u] = count of u'th user
	public int[] N_d;       // N_d[d] = count of d'th document
	public int[][] N_yz;    // N_yz[y][z] = count of y'th interest and z'th topic
		

	// private variables 
	private double b_on_W;
	private double au_on_Y;
	private double ad_on_Z;
	
	// parameter estimates
	public double[][] P_z_d; 
	public double[][] P_y_u; 

	
	public double[][] estimate(int[][][] w_uij, int[][] d_ui, int W, int D, long N, int Y, int Z, double au, double ad, double b, int burnin, int samples, int step) {
		
		this.w_uij = w_uij;
		this.d_ui = d_ui;
		this.W = W;
		this.U = w_uij.length;
		this.D = D;
		this.N = N;
		this.Y = Y;
		this.Z = Z;
		this.au = au;
		this.ad = ad;
		this.b = b;
		
		b_on_W = b/W;
		au_on_Y = au/Y;
		ad_on_Z = ad/Z;
		
		// initialise param matrices
		P_z_d= new double[Z][D];
		P_y_u = new double[Y][U];

		// initialize latent variable assignment and count matrices
		y_uij = new int[U][][];
		z_uij = new int[U][][];
		N_yu = new int[Y][U];
		N_zd = new int[Z][D];
		N_wyz = new int[W][Y][Z];
		N_u = new int[U];
		N_d = new int[D];
		N_yz = new int[Y][Z];
		for (int u=0; u<U; u++) {  // user u
			int I = w_uij[u].length;
			y_uij[u] = new int[I][];
			z_uij[u] = new int[I][];
			for (int i=0; i<I; i++) { // query i
				int J = w_uij[u][i].length;
				y_uij[u][i] = new int[J];
				z_uij[u][i] = new int[J];
				int d = d_ui[u][i];
				for (int j=0; j<J; j++) { // position j
					int y = (int) (Y * Math.random());
					int z = (int) (Z * Math.random());
					y_uij[u][i][j] = y;
					z_uij[u][i][j] = z;
					N_yu[y][u]++;
					N_zd[z][d]++;
					N_wyz[w_uij[u][i][j]][y][z]++;
					N_u[u]++;
					N_d[d]++;
					N_yz[y][z]++;
				}	
			}
		}
		
		// perform Gibbs sampling
		for (int iteration=0; iteration<(burnin+samples); iteration++) {
			for (int u=0; u<w_uij.length; u++) {  // user u
				for (int i=0; i<w_uij[u].length; i++) { // query i
					int d = d_ui[u][i];
					for (int j=0; j<w_uij[u][i].length; j++) { // position j
						
						int w = w_uij[u][i][j];
						int y = y_uij[u][i][j];
						int z = z_uij[u][i][j];
						
						// ** FIRST SAMPLE y 
						
						// remove last value  
						N_yu[y][u]--;
						N_wyz[w][y][z]--;
						N_yz[y][z]--;
						
						// calculate distribution
						double[] p = new double[Y];
						double total = 0;
						for (y=0; y<Y; y++) {
							p[y] = ( (N_wyz[w][y][z] + b_on_W)/(N_yz[y][z] + b) ) * (N_yu[y][u] + au_on_Y);
							total += p[y];
						}
						
						// resample 
						double val = total * Math.random();
						y = 0; while ((val -= p[y]) > 0) y++;
						
						// update latent variable and counts
						y_uij[u][i][j] = y;
						
						N_yu[y][u]++;

						// ** NOW SAMPLE z
						
						N_zd[z][d]--;
						
						// calculate distribution
						p = new double[Z];
						total = 0;
						for (z=0; z<Z; z++) {
							p[z] = ( (N_wyz[w][y][z] + b_on_W)/(N_yz[y][z] + b) ) * (N_zd[z][d] + ad_on_Z) ;
							total += p[z];
						}
						
						// resample 
						val = total * Math.random();
						z = 0; while ((val -= p[z]) > 0) z++;
						
						// update latent variable and counts
						z_uij[u][i][j] = z;
						
						N_zd[z][d]++;
						N_wyz[w][y][z]++;
						N_yz[y][z]++;
						
					}
				}	
			}	
			
			if(iteration >= burnin) {
				for (int u=0; u<U; u++)	for (int y=0; y<Y; y++)	P_y_u[y][u] += (N_yu[y][u] + au_on_Y)/(N_u[u] + au);
				for (int d=0; d<D; d++)	for (int z=0; z<Z; z++)	P_z_d[z][d] += (N_zd[z][d] + ad_on_Z)/(N_d[d] + ad);
				if (iteration%step==0) updateP_w_yz();
			}
			
			if (iteration%step==0) System.out.println("PTM3 (seq): iteration: "+iteration+", log-likelihood: "+logLikelihood());
			
		}
		
		// normalize parameter estimates
		for (int u=0; u<U; u++)	for (int y=0; y<Y; y++)	P_y_u[y][u] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_z_d[z][d] /= samples;
		normalizeAndSaveP_w_yz("P_w_yz.data");
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.saveMatrix(P_z_d,"P_z_d.data");
		TopicModelUtils.saveMatrix(P_y_u,"P_y_u.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");
		
		return estimateP_w_z();
	}	
		
	
	public double logLikelihood() {
		double ll = 0.0;
		for (int u=0; u<U; u++) {  // user u
			for (int i=0; i<w_uij[u].length; i++) { // query i
				int d = d_ui[u][i];
				for (int j=0; j<w_uij[u][i].length; j++) { // position j
					int y = y_uij[u][i][j];
					int z = z_uij[u][i][j];
					int w = w_uij[u][i][j];
					ll += Math.log( (N_wyz[w][y][z] + b_on_W)/(N_yz[y][z] + b) );
					ll += Math.log( (N_yu[y][u] + au_on_Y)/(N_u[u] + au) );
					ll += Math.log( (N_zd[z][d] + ad_on_Z)/(N_d[d] + ad) );
				}
			}
		}
		return ll;
	}
	
	
	int samplesP_w_yz = 0;
	public void updateP_w_yz() {
		for(int w = 0; w < W; w++) {
			// load previous values
			double[][] P_w_yz = (samplesP_w_yz == 0) ? new double[Y][Z] : TopicModelUtils.loadNonRectMatrix("temp_"+w+".data");
			// update
			for (int y=0; y<Y; y++) for (int z=0; z<Z; z++) P_w_yz[y][z] += (N_wyz[w][y][z] + b_on_W)/(N_yz[y][z] + b);
			// save
			TopicModelUtils.saveNonRectMatrix(P_w_yz, "temp_"+w+".data");
		}
		samplesP_w_yz++;
	}

	
	public void normalizeAndSaveP_w_yz(String filename) {
		// Let's save out the big matrix
		try {
			PrintWriter p = new PrintWriter(new FileOutputStream(filename));
			p.println(W); // rows
			p.println(Y); // columns
			p.println(Z); // depth
			for(int w = 0; w < W; w++) {
				double[][] P_w_yz = TopicModelUtils.loadNonRectMatrix("temp_"+w+".data");
				for (int y=0; y<Y; y++) {
					for (int z=0; z<Z; z++) {
						if (z>0) p.print(' '); 
						p.print(P_w_yz[y][z]/samplesP_w_yz);
					}	
					p.println();
				}	
				// and now we delete the temporary file...
				(new File("temp_"+w+".data")).delete();
			}
			p.close();
		}
		catch(Exception e) { e.printStackTrace(); }
	}
	
	
	public double[] estimateP_d() {
		double[] p_d = new double[D];
		for (int d=0; d<D; d++) p_d[d] = N_d[d]/(double) N;
		return p_d;
	}

	
	// this method is used just to find the top terms for each topic
	public double[][] estimateP_w_z() {
		double[][] p_w_z = new double[W][Z];
		// Here we "sort of" marginalize out y
		// The calculation probably insn't quite correct but
		// may not be a bad approximation and may be sufficient
		// for understanding the quality of the topics produced
		int[] N_z = new int[Z];
		for (int y=0; y<Y; y++) 
			for (int z=0; z<Z; z++) 
				N_z[z] += N_yz[y][z];
		for (int w=0; w<W; w++) 
			for (int z=0; z<Z; z++) {
				int N_wz = 0; 
				for (int y=0; y<Y; y++) N_wz += N_wyz[w][y][z];
				p_w_z[w][z] = (N_wz + b_on_W)/(N_z[z] + b);
			}	
		return p_w_z;
	}
	

	
}
