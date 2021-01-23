int _LENGTH(void* _ptr);
void GAUSSIAN(double VAR, double mu, double sigma);


int *a;
double *b;


void _CONSTRAINT()
{
	_LENGTH(a) >= 4 && _LENGTH(a) <= 6;
	
	a[0] + a[1] + a[2] + a[3] == 100;
	
	_LENGTH(b) >= 3 && _LENGTH(b) <= 4;
	
	GAUSSIAN(b[0], 0, 1);
	b[0] < b[1];
	b[1] < 1;
}
