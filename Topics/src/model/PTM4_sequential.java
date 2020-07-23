package model;

import utils.TopicModelUtils;


public class PTM4_sequential {
	
	public int X; // number of "term-groups"
	public int Y; // number of user interests
	public int Z; // number of document topics
	public int W; // vocabulary
	public int U; // number of users
	public int D; // number of documents (clicked urls)
	public long N; // total word occurrences
	
	public double au; // interest smoothing hyperparameter for users
	public double ad; // topic smoothing hyperparameter for documents 
	public double bx; // term-group smoothing hyperparamter  
	public double b; // term smoothing hyperparamter \beta 
	
	public int[][][] w_uij; // w_uij[u][i][j] = j'th word in the i'th query by the u'th user
	public int[][][] x_uij; // x_uij[u][i][j] = term-group assignment to j'th position in the i'th query by the u'th user
	public int[][][] y_uij; // y_uij[u][i][j] = interest assignment to j'th position in the i'th query by the u'th user
	public int[][][] z_uij; // z_uij[u][i][j] = topic assignment to j'th position in the i'th query by the u'th user
	public int[][] d_ui;    // d_ui[u][i] = document for i'th query by the u'th user
	
	public int[][] N_yu;    // N_yu[y][u] = count of y'th interest for the u'th user
	public int[][] N_zd;    // N_zd[z][d] = count of z'th topic for the d'th document
	public int[][][] N_xyz; // N_xyz[x][y][z] = count of x'th word-group for y'th interest and z'th topic
	public int[][] N_wx;    // N_wx[w][x] = count of w'th word for x'th word-group
	public int[] N_u;       // N_u[u] = count of u'th user
	public int[] N_d;       // N_d[d] = count of d'th document
	public int[][] N_yz;    // N_yz[y][z] = count of y'th interest and z'th topic
	public int[] N_x;       // N_x[x] = count of x'th word-group
	
	// private variables 
	private double b_on_W;
	private double bx_on_X;
	private double au_on_Y;
	private double ad_on_Z;
	
	// parameter estimates
	public double[][] P_z_d; 
	public double[][] P_y_u;
	public double[][][] P_x_yz;
	public double[][] P_w_x;
 
	
	public double[][] estimate(int[][][] w_uij, int[][] d_ui, int W, int D, long N, int X, int Y, int Z, double au, double ad, double bx, double b, int burnin, int samples, int step) {
		
		this.w_uij = w_uij;
		this.d_ui = d_ui;
		this.W = W;
		this.U = w_uij.length;
		this.D = D;
		this.N = N;
		this.X = X;
		this.Y = Y;
		this.Z = Z;
		this.au = au;
		this.ad = ad;
		this.bx = bx;
		this.b = b;
		
		b_on_W = b/W;
		bx_on_X = bx/X;
		au_on_Y = au/Y;
		ad_on_Z = ad/Z;
		
		// initialise param matrices
		P_z_d= new double[Z][D];
		P_y_u = new double[Y][U];
		P_x_yz = new double[X][Y][Z];
		P_w_x = new double[W][X];
		
		// initialize latent variable assignment and count matrices
		x_uij = new int[U][][];
		y_uij = new int[U][][];
		z_uij = new int[U][][];
		N_yu = new int[Y][U];
		N_zd = new int[Z][D];
		N_xyz = new int[X][Y][Z];
		N_wx = new int[W][X];
		N_u = new int[U];
		N_d = new int[D];
		N_yz = new int[Y][Z];
		N_x = new int[X];
		for (int u=0; u<w_uij.length; u++) {  // user u
			int I = w_uij[u].length;
			x_uij[u] = new int[I][];
			y_uij[u] = new int[I][];
			z_uij[u] = new int[I][];
			for (int i=0; i<I; i++) { // query i
				int J = w_uij[u][i].length;
				x_uij[u][i] = new int[J];
				y_uij[u][i] = new int[J];
				z_uij[u][i] = new int[J];
				int d = d_ui[u][i];
				for (int j=0; j<J; j++) { // position j
					int x = (int) (X * Math.random());
					int y = (int) (Y * Math.random());
					int z = (int) (Z * Math.random());
					x_uij[u][i][j] = x;
					y_uij[u][i][j] = y;
					z_uij[u][i][j] = z;
					N_yu[y][u]++;
					N_zd[z][d]++;
					N_xyz[x][y][z]++;
					N_wx[w_uij[u][i][j]][x]++;
					N_u[u]++;
					N_d[d]++;
					N_yz[y][z]++;
					N_x[x]++;
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
						int x = x_uij[u][i][j];
						int y = y_uij[u][i][j];
						int z = z_uij[u][i][j];
						
						// ** FIRST SAMPLE y 
						// remove last value  
						N_yu[y][u]--;
						N_xyz[x][y][z]--;
						N_yz[y][z]--;
						// calculate distribution
						double[] p = new double[Y];
						double total = 0;
						for (y=0; y<Y; y++) {
							p[y] = ( (N_xyz[x][y][z] + bx_on_X)/(N_yz[y][z] + bx) ) * (N_yu[y][u] + au_on_Y);
							total += p[y];
						}
						// resample 
						double val = total * Math.random();
						y = 0; while ((val -= p[y]) > 0) y++;
						// update latent variable and counts
						y_uij[u][i][j] = y;
						N_yu[y][u]++;

						// ** THEN SAMPLE z
						N_zd[z][d]--;
						// calculate distribution
						p = new double[Z];
						total = 0;
						for (z=0; z<Z; z++) {
							p[z] = ( (N_xyz[x][y][z] + bx_on_X)/(N_yz[y][z] + bx) ) * (N_zd[z][d] + ad_on_Z) ;
							total += p[z];
						}
						// resample 
						val = total * Math.random();
						z = 0; while ((val -= p[z]) > 0) z++;
						// update latent variable and counts
						z_uij[u][i][j] = z;
						N_zd[z][d]++;
						N_yz[y][z]++;

						// ** THEN SAMPLE x
						N_wx[w][x]--;
						N_x[x]--;
						// calculate distribution
						p = new double[X];
						total = 0;
						for (x=0; x<X; x++) {
							p[x] = ( (N_wx[w][x] + b_on_W)/(N_x[x] + b) ) * (N_xyz[x][y][z] + bx_on_X) ;
							total += p[x];
						}
						// resample 
						val = total * Math.random();
						x = 0; while ((val -= p[x]) > 0) x++;
						// update latent variable and counts
						x_uij[u][i][j] = x;
						N_xyz[x][y][z]++;
						N_wx[w][x]++;
						N_x[x]++;

					}
				}	
			}	
			
			if(iteration >= burnin) {
				for (int u=0; u<U; u++)	for (int y=0; y<Y; y++)	P_y_u[y][u] += (N_yu[y][u] + au_on_Y)/(N_u[u] + au);
				for (int d=0; d<D; d++)	for (int z=0; z<Z; z++)	P_z_d[z][d] += (N_zd[z][d] + ad_on_Z)/(N_d[d] + ad);
				for (int x=0; x<X; x++) for (int y=0; y<Y; y++) for (int z=0; z<Z; z++) P_x_yz[x][y][z] += (N_xyz[x][y][z] + bx_on_X)/(N_yz[y][z] + bx);
				for (int w=0; w<W; w++) for (int x=0; x<X; x++) P_w_x[w][x] += (N_wx[w][x] + b_on_W)/(N_x[x] + b);
			}
			if (iteration%step==0) System.out.println("PTM4 (seq): iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}
		
		// normalize parameter estimates
		for (int u=0; u<U; u++)	for (int y=0; y<Y; y++)	P_y_u[y][u] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_z_d[z][d] /= samples;
		for (int x=0; x<X; x++) for (int y=0; y<Y; y++) for (int z=0; z<Z; z++)	P_x_yz[x][y][z] /= samples;
		for (int w=0; w<W; w++) for (int x=0; x<X; x++)	P_w_x[w][x] /= samples;
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.saveMatrix(P_w_x,"P_w_x.data");
		TopicModelUtils.save3dMatrix(P_x_yz,"P_x_yz.data");
		TopicModelUtils.saveMatrix(P_z_d,"P_z_d.data");
		TopicModelUtils.saveMatrix(P_y_u,"P_y_u.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");
		
		return P_w_x;
	}	
	
	
	public double logLikelihood() {
		double ll = 0.0;		
		for (int u=0; u<U; u++) {  // user u
			for (int i=0; i<w_uij[u].length; i++) { // query i
				int d = d_ui[u][i];
				for (int j=0; j<w_uij[u][i].length; j++) { // position j
					int x = x_uij[u][i][j];
					int y = y_uij[u][i][j];
					int z = z_uij[u][i][j];
					int w = w_uij[u][i][j];
					ll += Math.log( (N_wx[w][x] + b_on_W)/(N_x[x] + b) );
					ll += Math.log( (N_xyz[x][y][z] + bx_on_X)/(N_yz[y][z] + bx) );
					ll += Math.log( (N_yu[y][u] + au_on_Y)/(N_u[u] + au) );
					ll += Math.log( (N_zd[z][d] + ad_on_Z)/(N_d[d] + ad) );
				}
			}
		}		
		return ll;
	}
	
	
	public double[] estimateP_d() {
		double[] p_d = new double[D];
		for (int d=0; d<D; d++)
			p_d[d] = N_d[d]/(double) N;
		return p_d;
	}

}
