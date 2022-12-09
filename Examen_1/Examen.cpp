////////////////////////////////Datos del alumno///////////////////////////////////////
/*
Nombre: Bruno González Lucero
Grupo: 5BV1
Materia: Visión Artificial
Profesor: Sanchez García Octavio
EXAMEN PRÁCTICO 1
*/

////////////////////////////////Importacion de librerias/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
/////////////////////////////////////////////////////////////////////////////

using namespace cv;
using namespace std;

vector<vector<float>> generateKernel(int kSize, int sigma) {
	float e = 3.1416;
	float pi = 2.72;
	int amountSlide = (kSize - 1) / 2;
	vector<vector<float>> v(kSize, vector<float>(kSize, 0));
	// si el centro es (0,0)
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float resultado = (1 / (2 * pi * sigma * sigma)) * pow(e, -((i * i + j * j) / (2 * sigma * sigma)));
			v[i + amountSlide][j + amountSlide] = resultado;
		}
	}
	return v;
}
float applyFilterToPix(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int rows = original.rows;
	int cols = original.cols;
	int amountSlide = (kSize - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float kTmp = kernel[i + amountSlide][j + amountSlide];
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = original.at<uchar>(Point(tmpX, tmpY));
				//cout << tmpX << " "<< tmpY << " "<< kTmp << endl;
			}

			sumFilter += (kTmp * tmp);
			sumKernel += kTmp;
		}
	}
	return sumFilter / sumKernel;
}
Mat applyFilterToMat(Mat original, vector<vector<float>> kernel, int kSize) {
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++) {
			filteredImg.at<uchar>(Point(i, j)) = uchar(applyFilterToPix(original, kernel, kSize, i, j));
		}
	}
	return filteredImg;
}

//Declaracion de funciones para el Kernel ///
float Gauss(int x, int y);
float** CrearKernelGauss(float** Kernel, int d);
vector<vector<float>> CrearKernelSobelX(int XY);
vector<vector<float>> CrearKernelSobelY(int XY);
float** LlenarKernel(float** Kernel, int d);
void ImprimirKernel(float** Kernel, int d);
void DestruirKernel(float** Kernel, int d);
Mat procesarMatriz(Mat imagen, int kernel, int sigma);
Mat ecualizado(Mat imagen);
void ImprimirSobel(vector<vector<float>> Sobel);
Mat AplicarSobel(Mat imagen, Mat Recipiente, vector<vector<float>> sobel, int XY);
Mat G_Valor(Mat sobelx, Mat sobely, Mat resultadoG);
Mat Umbral(Mat Operador, Mat Recipiente);


int main(int argc, char* argv[]) {
	float** Kernel = NULL;
	int Ksize = 7, sigma = 8;
	//Cargarmos imagen al programa
	imread("Lena.jpg");
	Mat image = imread("C:\\Users\\bgonz\\Documents\\MATLAB\\vision_artificial\\Proyectos\\Examen_1\\Lena.jpg");
	imshow("Imagen Original", image);
	int fila_original = image.rows;
	int columna_original = image.cols;//Lectura de la cantidad de columnas y filas
	printf("Dimensiones de la imagen de entrada: \n");
	printf("%d pixeles de largo \n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	Mat imagenGrisesNTSC(fila_original, columna_original, CV_8UC1);
	for (int i = 0; i < fila_original; i++)
	{
		for (int j = 0; j < columna_original; j++)
		{
			double azul = image.at<Vec3b>(Point(j, i)).val[0];  // B
			double verde = image.at<Vec3b>(Point(j, i)).val[1]; // G
			double rojo = image.at<Vec3b>(Point(j, i)).val[2];  // R

			// Conversion a escala de grises
			imagenGrisesNTSC.at<uchar>(Point(j, i)) = uchar(0.299 * rojo + 0.587 * verde + 0.114 * azul);
		}
	}

	// Agregamos bordes
	Mat image2 = procesarMatriz(image, Ksize, sigma);
	imshow("Imagen con filas extra", image2);
	fila_original = image2.rows;
	columna_original = image2.cols;
	printf("Dimensiones de la imagen con bordes adcicionales: \n");
	printf("%d pixeles de largo\n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	// Creamos Kernel
	Kernel = CrearKernelGauss(Kernel, Ksize);
	ImprimirKernel(Kernel, Ksize);
	vector<vector<float>> kernel = generateKernel(Ksize, sigma);

	//Aplicamos Kernel a la imagen
	Mat filtrada = applyFilterToMat(imagenGrisesNTSC, kernel, Ksize);
	imshow("Imagen Filtrada", filtrada);
	fila_original = filtrada.rows;
	columna_original = filtrada.cols;
	printf("\nDimensiones de la imagen con bordes adcicionales filtrada: \n");
	printf("%d pixeles de largo\n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	//Ecualizamos el histograma
	Mat ecualizada = ecualizado(filtrada);
	imshow("Imagen Ecualizada", ecualizada);
	fila_original = ecualizada.rows;
	columna_original = ecualizada.cols;
	printf("\nDimensiones de la imagen ya ecualizada: \n");
	printf("%d pixeles de largo\n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	// Creamos Kernel de Sobel
	vector<vector<float>> SobelX = CrearKernelSobelX(1);
	vector<vector<float>> SobelY = CrearKernelSobelY(1);
	ImprimirSobel(SobelX);
	ImprimirSobel(SobelY);
	//Aplicamos SobelX y SobelY
	Mat sobelx;
	Mat sobely;
	sobelx = AplicarSobel(ecualizada, sobelx, SobelX, 1);
	sobely = AplicarSobel(ecualizada, sobely, SobelY, 0);

	//Procedimiento para mostrar individualmente SobelX y SobelY
	imshow("Imagen de Transicion Con Sobel en X", sobelx);
	fila_original = sobelx.rows;
	columna_original = sobelx.cols;
	printf("\nDimensiones de la imagen con Sobel en X: \n");
	printf("%d pixeles de largo\n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	imshow("Imagen de Transicion Con Sobel en Y", sobely);
	fila_original = sobely.rows;
	columna_original = sobely.cols;
	printf("\nDimensiones de la imagen con Sobel en Y: \n");
	printf("%d pixeles de largo\n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	//Aplicamos |G| con SobelX y SobelY
	Mat resultadoG;
	resultadoG = G_Valor(sobelx, sobely, resultadoG);
	imshow("Imagen |G|", resultadoG);
	fila_original = resultadoG.rows;
	columna_original = resultadoG.cols;
	printf("\nDimensiones de la imagen con Sobel |G|: \n");
	printf("%d pixeles de largo\n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	//Umbralado Canny
	Mat umbralado;
	umbralado = Umbral(resultadoG, umbralado);
	imshow("Operador Canny: Umbralado", umbralado);
	fila_original = umbralado.rows;
	columna_original = umbralado.cols;
	printf("\nDimensiones de la imagen umbralada: \n");
	printf("%d pixeles de largo\n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

	//Liberamos Memoria
	DestruirKernel(Kernel, Ksize);
	waitKey(0);
	return(0);
}

/// Funciones para el Kernel ///

//Recibe como parámetros un apuntador doble vacío (Kernel) y su tamaño, para generar posteriormente un arreglo y 
//llenándolo con la función LlenarKernel()
float** CrearKernelGauss(float** Kernel, int d)
{
	int i = 0, j = 0;

	Kernel = (float**)malloc(d * sizeof(float*));

	for (i = 0; i < d; i++)
		Kernel[i] = (float*)malloc(d * sizeof(float));

	Kernel = LlenarKernel(Kernel, d);
	return (Kernel);
}

//Recibe como parámetro un número y genera la matriz de sobel para los valores de X, regresandola como vector 
//de dos dimensiones
vector<vector<float>> CrearKernelSobelX(int XY)
{
		vector<vector<float>> Sobel
		{
			{-1, 0, 1},
			{-2, 0, 2},
			{-1, 0, 1}
		};

	return Sobel;
}

//Recibe como parámetro un número y genera la matriz de sobel para los valores de Y, regresandola como vector 
//de dos dimensiones
vector<vector<float>> CrearKernelSobelY(int XY)
{
	vector<vector<float>> Sobel
	{
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};

	return Sobel;
}

//Recibe como parámetros un apuntador doble vacío (Kernel) y su tamaño y lo rellena a partir de la división del
//kernel en 4 cuadrantes usando la formula del kernel gaussiano
float** LlenarKernel(float** Kernel, int d)
{
	//i representa el eje Y y j representa el eje X 

	int i = 0, j = 0;
	int x = 0, y = 0;

	for (i = d / 2; i < d; i++)	//Llenamos el cuadrante I
	{
		for (j = d / 2; j < d; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = 0;
		y += 1;
	}
	x = 0;
	y = -(d / 2);

	for (i = 0; i < d / 2; i++)	//Llenamos el cuadrante II
	{
		for (j = d / 2; j < d; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = 0;
		y += 1;
	}
	x = -(d / 2);
	y = -(d / 2);

	for (i = 0; i < d / 2; i++)	//Llenamos el cuadrante III
	{
		for (j = 0; j < d / 2; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = -(d / 2);
		y += 1;
	}
	x = -(d / 2);
	y = 0;

	for (i = d / 2; i < d; i++)	//Llenamos el cuadrante IV
	{
		for (j = 0; j < d / 2; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = -(d / 2);
		y += 1;
	}
	return(Kernel);
}

//Recibe como parámetros un apuntador doble (Kernel) y su tamaño para proceder a mostrarlo en pantalla
void ImprimirKernel(float** Kernel, int d)
{
	int i = 0, j = 0;

	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
			printf("%.3f\t", Kernel[i][j]);
		printf("\n");
	}
}

//Recibe como parámetros un apuntador doble (Kernel) y su tamaño para liberar el espacio de memoria
//correspondiente al kernel
void DestruirKernel(float** Kernel, int d)
{
	int i = 0, j = 0;
	for (i = 0; i < d; i++)
	{
		free(Kernel[i]);
		Kernel[i] = NULL;
	}
	free(Kernel);
}

//Recibe como parámetros un par de coordenadas con las que se aplica el método correspondiente para
//obtener su valor después de applicar el kernel de Gauss
float Gauss(int x, int y)
{
	float pi = 3.1416, e = 2.71828;
	float sigma = 1, F_1 = 0, F_2 = 0, potencia = 0;
	float valor = 0;

	F_1 = (1) / (2 * pi * pow(sigma, 2));
	potencia = (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2));
	F_2 = pow(e, 0 - potencia);
	valor = F_1 * F_2;

	return(valor);
}

//Recibe como parámetros la imagen en formato Mat, el tamaño del kernel y el sigma asignado a la función
//prar transformar a blanco y negro y añadir los bordes antes de la aplicación del filtro gaussiano
Mat procesarMatriz(Mat imagen, int kernel, int sigma) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	int exceso = (kernel - 1);

	Mat grises(rows + exceso, cols + exceso, CV_8UC1);
	Mat grande(rows + exceso, cols + exceso, CV_8UC1);
	double rojo, azul, verde, gris_p;

	for (int i = 0; i < rows + exceso; i++) {
		for (int j = 0; j < cols + exceso; j++) {

			if (i >= rows || i < exceso) { // >=
				grande.at<uchar>(Point(j, i)) = uchar(0);
				//cout << "entra\n";


			}
			else if (j >= cols || j < exceso) { //nadamas le cambie por >=, ya que toma en cuenta el 0
				grande.at<uchar>(Point(j, i)) = uchar(0);
				//cout << "entra\n";
			}
			else {
				azul = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[0];
				//verde la segunda
				verde = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[1];
				//roja la tercer
				rojo = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[2];

				//el valor de gris promediado lo obtenemos sumando cada valor de 
				//rojo, verde y azul sobre 3
				gris_p = (azul + verde + rojo) / 3;

				grande.at<uchar>(Point(j, i)) = uchar(gris_p);
			}
			//azul = image.at<Vec3b>(Point(j, i)).val[0];
			//verde la segunda
			//verde = image.at<Vec3b>(Point(j, i)).val[1];
			//roja la tercer
			//rojo = image.at<Vec3b>(Point(j, i)).val[2];



			//grande.at<uchar>(Point(j, i)) = uchar(gris_p); //uchar es un valor de 8 bits

		}
	}
	return(grande);
}

//Recibe como parámetro la imágen resultado de aplicar el filtro gaussiano para ecualizar el histograma
//y devolver el resultado
Mat ecualizado(Mat imagen)
{
	Mat nuevo;
	equalizeHist(imagen, nuevo);
	return (nuevo);
}

//Recibe como parámetro el Kernel de sobel como un vector de dos dimensiones para mostrarlo en pantalla
void ImprimirSobel(vector<vector<float>> Sobel)
{
	cout << "\n";
	for (vector<float> ss : Sobel) {
		for (int sasas : ss) {
			cout << sasas << " ";
		}
		cout << "\n";
	}
}

//Recibe como parámetros la matriz de la imagen resultado de aplicar el filtro gaussiano, una matriz vacía
//que recibirá los valores, el arreglo de vectores correspondiente a un filtro de sobel y un número que
//determina si se aplica la función para SobelX o SobelY
Mat AplicarSobel(Mat imagen, Mat Recipiente, vector<vector<float>> sobel, int XY)
{
	if (XY == 1) {
		//SobelX
		Sobel(imagen, Recipiente, CV_8U, 1, 0, 1, 1, 0, BORDER_DEFAULT);
	}
	else {
		//SobelY
		Sobel(imagen, Recipiente, CV_8U, 0, 1, 1, 1, 0, BORDER_DEFAULT);
	}
	return Recipiente;
}

//Recibe como parámetros las imágenes con los filtros de sobel en X y Y aplicados junto con una matriz
//vacía que recibirá los valores de operar ambas imagenes
Mat G_Valor(Mat sobelx, Mat sobely, Mat resultadoG)
{
	Mat abs_sobelx, abs_sobely;
	convertScaleAbs(sobelx, abs_sobelx);
	convertScaleAbs(sobely, abs_sobely);

	addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0, resultadoG);

	return resultadoG;
}

//Recibe como parámetros la imágen con |G| aplicado y una matriz vacía que recibirá los valores del
//umbralado después de ser aplicado a la imagen |G|
Mat Umbral(Mat Operador, Mat Recipiente)
{
	int umbral_valor = 7;
	int max_binary_value = 255;
	int umbral_tipo = 0;
	threshold(Operador, Recipiente, umbral_valor, max_binary_value, umbral_tipo);
	return Recipiente;
}
