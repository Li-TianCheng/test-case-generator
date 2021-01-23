int _LENGTH(void* _ptr);
void GAUSSIAN(double VAR, double mu, double sigma);


typedef struct 
{
	double x;
	double y;
} A;

typedef struct 
{
	struct A a;
	int z;
} B;


struct A a;
struct B b;
int c;


void _CONSTRAINT()
{
	a.x > 5 && a.x < 10;
	a.x + a.y < 20;
	b.a.x + b.z > 10;
	b.a.x + b.z < 20;
	c == 4;
}
