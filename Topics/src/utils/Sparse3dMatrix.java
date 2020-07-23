package utils;

import java.util.ArrayList;

public class Sparse3dMatrix {
		
	public ArrayList<Integer> x  = new ArrayList<Integer>();
	public ArrayList<Integer> y  = new ArrayList<Integer>();
	public ArrayList<Integer> z  = new ArrayList<Integer>();
	public ArrayList<Double> val = new ArrayList<Double>();
	
	public Sparse3dMatrix(int size) {
		x  = new ArrayList<Integer>(size);
		y  = new ArrayList<Integer>(size);
		z  = new ArrayList<Integer>(size);
		val = new ArrayList<Double>(size);		
	}
	
	public void add(int i, int j, int k, double v) {
		x.add(i);
		y.add(j);
		z.add(k);
		val.add(v);
	}
	
}
