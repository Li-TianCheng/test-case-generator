
int _LENGTH(void* _ptr);
void GAUSSIAN(double VAR, double mu, double sigma);



int a[10][5];
int b[5];


void _CONSTRAINT()
{
	a[0][1] != 5;
	a[1][0] * a[1][1] == 20;
	b[0] < b[1];
	b[1] < b[2];
	b[2] < b[3];
	b[3] < b[4];
	b[0] >= -10;
	b[4] <= 10;

}

