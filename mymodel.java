
package model;

import data.LabeledReviews;


public class mymodel {

	public static int Z; // number of topics
	public int W; // vocabulary
	public static int D; // number of documents
	private static long N; // total word occurrences
	private static int L ; // labels
	private static int S ; // number of sentiment

	private  long[][] Nd_z_l ; // Count document level topic and label
	private  long[] Nd_l ; // Count document level label
	private  long[][][] Nws_s_zl ; //Count word level sentiment topic and label
	private  long[][] Nws_zl ; // Count word level topic and label
	private  long[][][] Nws_w_zs; //count word wi with sentiment s topic z
	private long[][]  Nws_zs ; //count words with topic z sentiment s
	private long[][] Nwt_w_z ; //count word w with topic z 
	private long[] Nwt_z ; //count words with topic z
	
	
	
	private int[][] is_di; // Switch for dth document ith word
	private int[][] sentiment ; //Sentiment of dth document ith word

	public int[] label ; // label of dth document
	public int[] topic ; //topic of dth document

	public double[][] P_z_l ;
	public double[][][] P_s_zl ;
	public double[][] P_w_z ;
	public double[][][] P_w_zs ;
	
	
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document

	public int num_samples;

	public void estimate(int[] label, int[][] w_di, String[] l_w, int W, int D, int Z, int L, int S ,  int burnIn, int samples, int step) {
		System.out.println("Estimation started!");
		
		
		
		
		this.w_di = w_di;
		this.label= label ;
		this.Z = Z;
		this.L = L ;
		this.S = S ;
		this.W = W;
		this.D = D;

		

		P_z_l = new double[Z][L] ;
		P_s_zl = new double[S][Z][L] ;
		P_w_z = new double[W][Z] ;
		P_w_zs = new double[W][Z][S] ;
		
		
		
		
		
//		b_wz = loadbetapriors(lr, beta_prior);
//		a_z = loadalphapriors(alpha_prior);

		num_samples=0 ;
		Nd_z_l= new long[Z][L] ;
		Nd_l = new long[L] ;
		
		Nws_s_zl = new long[S][Z][L];
		Nws_zl = new long[Z][L] ;
		
		is_di = new int[D][500] ;
		sentiment = new int[D][500] ;
		topic = new int[D] ;
		
		Nws_w_zs = new long[W][Z][S] ;
		Nws_zs = new long[Z][S] ;
		
		Nwt_w_z = new long[W][Z] ;
		Nwt_z = new long[Z] ;

		for(int d=0;d<D;d++){
			topic[d]= (int) (Math.random()*Z) ;
		}
		for(int d=0;d<D;d++){
			for(int i=0;i<w_di[d].length;i++){
				is_di[d][i] = (int) (Math.random()*2) ;
				if(is_di[d][i]==1)
					sentiment[d][i]= (int) (Math.random()*S) ;
				else
					sentiment[d][i] =-1 ;
			}
		}


		for(int d=0;d<D;d++){
			Nd_z_l[topic[d]][label[d]]++ ;
			Nd_l[label[d]]++ ; 
		}
		for(int d=0;d<D;d++){
			int z =topic[d] ;
			int l =label[d] ;
			for(int i=0;i<w_di[d].length;i++){
				long w = w_di[d][i];				
				int is = is_di[d][i];				
				int s = sentiment[d][i];
				
				if(is==1){
					Nws_s_zl[s][z][l]++ ;
					Nws_w_zs[(int) w][z][s]++ ;
					Nws_zs[z][s]++  ;
					Nws_zl[z][l]++; 
				}
				else {
					Nwt_z[z]++  ;
					Nwt_w_z[(int)w][z]++ ;
				}

			}
			
			

		}



		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples*step; iteration++) {
			
			
			
			for (int d=0; d<D; d++) { // document d
				
				if (w_di[d] == null)
					continue;
				int l=label[d];
				int z=topic[d] ;
				//Estimate probability of z/l
				double[] p = new double[Z] ;
				Nd_z_l[z][l]--;
				Nd_l[l]-- ;
				for(int i=0;i<w_di[d].length;i++){
					int w = w_di[d][i] ;
					int s = sentiment[d][i] ;
					
					if(is_di[d][i]==1){
						Nws_s_zl[s][z][l]-- ;
						Nws_zl[z][l]-- ;
						Nws_w_zs[w][z][s]-- ;
						Nws_zs[z][s]-- ;
						
					
					}else{
						Nwt_w_z[w][z]-- ;
						Nwt_z[z]-- ;
					}
				}



				for(int zi=0;zi<Z;zi++){

					p[zi] = Nd_z_l[zi][l]/Nd_l[l] ;

				}
				double val = Math.random() ;
				z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic
				
				Nd_z_l[z][l]++;
				Nd_l[l]++ ;
				
				
				for(int i=0;i<w_di[d].length;i++){
					int s= sentiment[d][i] ;
					int w =w_di[d][i] ;
					if(is_di[d][i]==1){
						Nws_s_zl[s][z][l]++ ;
						Nws_zl[z][l]++ ;
						Nws_w_zs[w][z][s]++ ;
						Nws_zs[z][s]++ ;
					}else {
						Nwt_w_z[w][z]++ ;
						Nwt_z[z]++ ;
					}
				}


				for (int i=0; i<w_di[d].length; i++) { // position i
					// Aadi: Go over each word of all documents
					
					int w = w_di[d][i];				// Aadi: Which word is this?
					int is = is_di[d][i];				//whether it is a topic word or sentiment word
					int s = sentiment[d][i] ; 							//sentiment of word
					if(is==0){
						Nwt_w_z[w][z]-- ;
						Nwt_z[z]-- ;
					}else if(is==1){
						Nws_s_zl[s][z][l]-- ;
						Nws_zl[z][l]-- ;
						Nws_w_zs[w][z][s]-- ;
						Nws_zs[z][s]-- ;
					}
					val = Math.random()*2 ;
					if(val<1)
						is_di[d][i]=0 ;
					else
						is_di[d][i]=1 ;
					is= is_di[d][i] ;
					
					if(is==0){
						Nwt_w_z[w][z]++ ;
						Nwt_z[z]++ ;
					}else if(is==1){
						Nws_s_zl[s][z][l]++ ;
						Nws_zl[z][l]++ ;
						Nws_w_zs[w][z][s]++ ;
						Nws_zs[z][s]++ ;
					}
				
					

					if(is==1){
						// Sample s ~ P(s|z,l,is=1). P(w|s,z,is=1)
						Nws_s_zl[s][z][l]-- ;
						Nws_zl[z][l]-- ;
						Nws_w_zs[w][z][s]-- ;
						Nws_zs[z][s]-- ;

						p = new double[S];

						double p_w_zs; 
						double p_s_zl; 
						
						double total =0 ;
						for( s=0;s<S;s++){
							p_w_zs=Nws_w_zs[w][z][s]/Nws_zs[z][s] ;
							p_s_zl = Nws_s_zl[s][z][l]/Nws_zl[z][l] ;
							p[s] = p_w_zs*p_s_zl ;
							total+=p[s] ;
						}
						for( s=0;s<S;s++)
							p[s]/=total ;
						val = Math.random();
 						s=0;while ((val -= p[s]) > 0) s++;
 						
 						Nws_s_zl[s][z][l]++ ;
						Nws_zl[z][l]++ ;
						Nws_w_zs[w][z][s]++ ;
						Nws_zs[z][s]++ ;
					
					}

				
				}	
			}	

			// update parameter estimates
			if (iteration >= burnIn && (iteration-burnIn)%step==0) {	

					// P(z|l)
					for(int l=0;l<L;l++){
						for(int z=0;z<Z;z++){
							P_z_l[z][l] += (double)Nd_z_l[z][l]/Nd_l[l] ;
						}
					}
					//P(s|z,l,is=1)
					for(int s=0;s<S;s++){
						for(int l=0;l<L;l++){
							for(int z=0;z<Z;z++){
								P_s_zl[s][z][l]+= (double)Nws_s_zl[s][z][l]/Nws_zl[z][l] ;
							}
						}
					}
					//P(w|z,is=0)
					for(int w =0;w<W;w++){
						for(int z=0;z<Z;z++){
							P_w_z[w][z]+= (double)Nwt_w_z[w][z]/Nwt_z[z] ;
						}
					}
					//P(w|s,z,is=1)
					for(int w=0;w<W;w++){
						for(int s=0;s<S;s++){
							for(int z=0;z<Z;z++){
								P_w_zs[w][z][s]+= (double)Nws_w_zs[w][z][s]/Nws_zs[z][s] ; 
							}
						}
					}
					
					
			}

			int hest_step = 0;
			
			if (hpestimate)
			{
				//fill
			}


		}
		
		//Normalize P(z|l)
		for(int l=0;l<L;l++){
			double totalsum =0 ;
			for(int z=0;z<Z;z++){
				totalsum+=P_z_l[z][l] ;
			}
			for(int z=0;z<Z;z++){
				P_z_l[z][l]/=totalsum ;
			}
			
		}
		//Normalize P(s/z,l,is=1)
		for(int z=0;z<Z;z++){
			for(int l=0;l<L;l++){
				double totalsum =0 ;
				for(int s=0;s<S;s++){
					totalsum+= P_s_zl[s][z][l] ;
				}
				for(int s=0;s<S;s++){
					P_s_zl[s][z][l]/=totalsum  ;
				}
				
			}
		}
		// Normalize P(w/z,is=0)
		for(int z=0;z<Z;z++){
			double totalsum=0 ;
			for(int w=0;w<W;w++){
				totalsum+=P_w_z[w][z] ;
			}
			for(int w=0;w<W;w++){
				P_w_z[w][z]/=totalsum ;
			}
		}
		// Normalize P(w/s,z,is=1)
		for(int z=0;z<Z;z++){
			for(int s=0;s<S;s++){
				double total= 0;
				for(int w=0;w<W;w++){
					total+=P_w_zs[w][z][s] ;
				}
				for(int w=0;w<W;w++){
					P_w_zs[w][z][s]/=total ;
				}
				
				
			}
		}
		
	}








	public static long getN() {
		return N;
	}

	public static void setN(long n) {
		N = n;
	}
	
}