package utils;



/*************************************************************************
 *  Compilation:  javac Gamma.java
 *  Execution:    java Gamma 5.6
 *  
 *  Reads in a command line input x and prints Gamma(x) and
 *  log Gamma(x). The Gamma function is defined by
 *  
 *        Gamma(x) = integral( t^(x-1) e^(-t), t = 0 .. infinity)
 *
 *  Uses Lanczos approximation formula. See Numerical Recipes 6.1.
 *
 * 
 *
 *************************************************************************/

public class Gamma {

	static double logGamma(double x) {
		double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
		double ser = 1.0 + 76.18009173    / (x + 0)   - 86.50532033    / (x + 1)
				+ 24.01409822    / (x + 2)   -  1.231739516   / (x + 3)
				+  0.00120858003 / (x + 4)   -  0.00000536382 / (x + 5);
		return tmp + Math.log(ser * Math.sqrt(2 * Math.PI));
	}
	public static double gamma(double x) { return Math.exp(logGamma(x)); }

	public static double digamma2(double x) {
		double result = 0, xx, xx2, xx4;
		assert(x > 0);
		for ( ; x < 7; ++x)
			result -= 1/x;
		x -= 1.0/2.0;
		xx = 1.0/x;
		xx2 = xx*xx;
		xx4 = xx2*xx2;
		result += Math.log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
		return result;
	}

	
	public static double digamma(double x)
	{
		double h = x * 1e-8;
		double lg_x = logGamma(x);
		double lg_xh = logGamma(x+h);
		double derivative = (lg_xh - lg_x) / h;

		return derivative;
	}
	
	public static double digamma3(double x) {
		double GAMMA = 0.577215664901532860606512090082;
		double GAMMA_MINX = 1.e-12;
		double DIGAMMA_MINNEGX = -1250;
		double C_LIMIT = 49;
		double S_LIMIT = 1e-5;
	    double value = 0;

	    while (true){

	        if (x >= 0 && x < GAMMA_MINX) {
	            x = GAMMA_MINX;
	        }
	        if (x < DIGAMMA_MINNEGX) {
	            x = DIGAMMA_MINNEGX + GAMMA_MINX;
	            continue;
	        }
	        if (x > 0 && x <= S_LIMIT) {
	            return value + -GAMMA - 1 / x;
	        }

	        if (x >= C_LIMIT) {
	            double inv = 1 / (x * x);
	            return value + Math.log(x) - 0.5 / x - inv
	                    * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252));
	        }

	        value -= 1 / x;
	        x = x + 1;
	    }

	}
	public static void main(String[] args) { 
		double x = Double.parseDouble(args[0]);
		System.out.println("Gamma(" + x + ") = " + gamma(x));
		System.out.println("log Gamma(" + x + ") = " + logGamma(x));
		System.out.println("Digamma(" + x + ") = " + digamma(x));
		System.out.println("Digamma2(" + x + ") = " + digamma2(x));
		System.out.println("Digamma2(" + x + ") = " + digamma3(x));
	}

}
