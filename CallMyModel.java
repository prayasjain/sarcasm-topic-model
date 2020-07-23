package helloworld;


import java.io.IOException;

public class CallMyModel {
	
	private static String[] map_id_word ; //Vocubulary of words
	private static int W=0 ; // length of vocabulary 
	private static double[] is_w ;// probability that the word is an issue word is=0
	private static HashMap<String, Integer> hm ;
	private static int[][] positive,negative,sarcastic,w_di ;
	private static int[] label ;
	private static int Z =30,L=3,S=2,burnIn=1000,samples=100,step=10 ;

	
	private static void test_fun(){
		System.out.println(W);
		for(int i=0;i<10;i++){
			System.out.println(map_id_word[i]);
			System.out.println(is_w[i]);
		}
	}
	
	public static void main(String[] args){
		try{
			
			Create_vocab_switch() ; // Will fill values of vocab_size, map_id_word, is_w
			
			//Loading tweets and mapping each word to ID

			hm= new HashMap<String, Integer>() ;
			for (int i=0;i<map_id_word.length;i++){
				hm.put(map_id_word[i],i) ;
			}
			String fpath = "Path to positive" ;
			positive = loadTweets(fpath) ;
			fpath = "path to negative" ;
			negative = loadTweets(fpath) ;
			fpath = "path to sarcastic" ;
			sarcastic = loadTweets(fpath) ;

			Integer[] index = new Integer[positive.length + negative.length + sarcastic.length] ;
			for(int i=0;i<positive.length + negative.length + sarcastic.length;i++)
				index[i]=i ;
			Collections.shuffle(Arrays.asList(index)) ;
			
			w_di = new int[positive.length + negative.length + sarcastic.length][2] ;
			label = new int[positive.length + negative.length + sarcastic.length] ;

			for(int i=0;i<index.length;i++){
				if(index[i]<positive.length){
					label[i] = 0 ;
					w_di[i] = new int[positive[index[i]].length] ;
					for(int j=0;j<w_di[i].length;j++){
						w_di[i][j]= positive[index[i]][j] ;
					}

				}else if(index[i]<(positive.length+negative.length)){
					label[i] =1 ;
					w_di[i] = new int[negative[index[i]-positive.length].length] ;
					for(int j=0;j<w_di[i].length;j++){
						w_di[i][j]= negative[index[i]-positive.length][j] ;
					}

				}else {
					label[i]=2 ;
					w_di[i] = new int[sarcastic[index[i]-positive.length-negative.length].length] ;
					for(int j=0;j<w_di[i].length;j++){
						w_di[i][j]= sarcastic[index[i] - positive.length - negative.length][j] ;
					}

				}
			}

			mymodel model = new mymodel() ;
			model.estimate(label,w_di,map_id_word,W,w_di.length,Z,L,S,burnIn,samples,step) ;

		}
		catch(IOException e){
			System.out.println(e.getMessage());
		}
		test_fun() ;
	}

	private static int[][] loadTweets(String fpath) {
		try{
			ReadFile file = new ReadFile(fpath) ;
			String[] fdata = file.OpenFile() ;
			int[][] tweets = new int[fdata.length][2] ;
			int[][] tweets_mapped = new int[fdata.length][2] ;
			for(int i=0;i<fdata.length;i++){
				String[] words = fdata[i].split(" ") ;
				tweets[i]= new int[words.length] ;
				int count =0 ;
				for(int j=0;j<words.length;j++){
					String w = words[j] ;
					if(hm.containsKey(w)){
						tweets[i][count++] = hm.get(w) ;
					}
				}
				tweets_mapped[i]= new int[count] ;
				for(int j=0;j<count;j++){
					tweets_mapped[i][j]= tweets[i][j] ;
				}
				return tweets_mapped ;
			}
		}catch( IOException e){
			System.out.println("Load Tweets");
			System.out.println(e.getMessage());
			return null ;
		}


	}

	private static  void Create_vocab_switch() throws IOException{
		String fpath = "/home/prayas/Desktop/topic-model-data/switch-probabilities.txt" ; //path to switch probab
		ReadFile file =new ReadFile(fpath) ;
		String[] fdata = file.OpenFile()  ;
		W=fdata.length-1 ;
		map_id_word = new String[W] ;
		is_w = new double[W] ;
		for(int i=1;i<W+1;i++){
			String[] splitted= fdata[i].split("\t") ;
			map_id_word[i-1]=splitted[0] ;
			is_w[i-1] = Double.parseDouble(splitted[1]) ;
		}
		
	}
	
}
