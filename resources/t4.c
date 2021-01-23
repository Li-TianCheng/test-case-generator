
int _LENGTH(void* _ptr);
void GAUSSIAN(double VAR, double mu, double sigma);

int *a;
double b[1];


void _CONSTRAINT()
{
	_LENGTH(a) >= 4 && _LENGTH(a) <= 6;
	
	a[0] > 100 || a[0] < 0;
	a[0] + b[0] < 1;
}
