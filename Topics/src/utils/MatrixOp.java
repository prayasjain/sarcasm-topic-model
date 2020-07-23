package utils;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class MatrixOp {
	
	public static void main(String args[]){
	double[][] matrixData = { {1d,2d,3d}, {2d,5d,3d}};
	RealMatrix m = MatrixUtils.createRealMatrix(matrixData);

	// One more with three rows, two columns, this time instantiating the
	// RealMatrix implementation class directly.
	double[][] matrixData2 = { {1d,2d}, {2d,5d}, {1d, 7d}};
	RealMatrix n = new Array2DRowRealMatrix(matrixData2);

	// Note: The constructor copies  the input double[][] array in both cases.

	// Now multiply m by n
	RealMatrix p = m.multiply(n);
	matrixData = p.getData();
	for (int i=0; i<2;i++)
	{	for(int j=0;j<2;j++)
			System.out.print(matrixData[i][j]+" ");
		System.out.println();
	}
	
	
	// Invert p, using LU decomposition
	RealMatrix pInverse = new LUDecomposition(p).getSolver().getInverse();
	matrixData = pInverse.getData();
	for (int i=0; i<2;i++)
	{	for(int j=0;j<2;j++)
			System.out.print(matrixData[i][j]+" ");
		System.out.println();
	}

	}
}
