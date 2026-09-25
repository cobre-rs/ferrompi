struct ldi { long double v; int i; };
struct li { long v; int i; };
char size_ldi[sizeof(struct ldi)];
char align_ldi[_Alignof(struct ldi)];
char size_li[sizeof(struct li)];
