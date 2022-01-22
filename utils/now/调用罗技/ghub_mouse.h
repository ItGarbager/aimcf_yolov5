//fork from https://github.com/ekknod/logitech-cve
#ifndef MOUSE_H
#define MOUSE_H


#define GHUB_API __declspec(dllexport)
typedef int BOOL;

#ifdef __cplusplus
extern "C" {
#endif

	BOOL GHUB_API Agulll(void);  // mouse_open
	void GHUB_API Shwaji(void);  // mouse_close
	void Check(char button, char x, char y, char wheel);  // mouse_move
	void GHUB_API Mach_Move(int x, int y);  // moveR
	void GHUB_API Leo_Kick(char button);  // press
	void GHUB_API Niman_years();  // release
	void GHUB_API Mebiuspin(char wheel);  // scroll

#ifdef __cplusplus
}
#endif

#endif
