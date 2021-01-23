int _LENGTH(void* ptr);
void GAUSSIAN(double mu, double sigma, double VAR);

typedef struct {
    int a;
    int b;
    double c;
    int* d;
} S1;

int a;
int b[3];
struct S1 s[2];

void _CONSTRAINT()
{
    a > 5 && a < 10;
    b[1] > b[2] || b[0] < b[1];
    a + b[0] != s[0].a;
    _LENGTH(s[0].d) > 6 && _LENGTH(s[0].d) < 10;
    _LENGTH(s[1].d) > 6 && _LENGTH(s[1].d) < 10;
    s[0].d[6] + s[1].d[6] == s[0].b * s[1].b;
    GAUSSIAN(s[0].c, 1.0, 1.0);
}
