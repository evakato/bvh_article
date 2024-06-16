#pragma once

namespace Tmpl8
{

// application class
class WhittedApp : public TheApp
{
public:
	// game flow methods
	void Init();
	void AnimateScene();
	float3 Trace( Ray& ray, int rayDepth = 0 );
	void TraceAVX(RayAVX& ray, __m256* r, __m256* g, __m256* b, int rayDepth = 0);
	//void TraceAVX(RayAVX& ray, float3* albedo, int rayDepth = 0);
	//void WhittedApp::TraceAVX(RayAVX& ray, float3* rgb, int rayDepth = 0);
	void Tick( float deltaTime );
	void Shutdown() { /* implement if you want to do something on exit */ }
	// input handling
	void MouseUp( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseDown( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove( int x, int y ) { mousePos.x = x, mousePos.y = y; }
	void MouseWheel( float y ) { /* implement if you want to handle the mouse wheel */ }
	void KeyUp( int key ) { /* implement if you want to handle keys */ }
	void KeyDown( int key ) { /* implement if you want to handle keys */ }
	// data members
	int2 mousePos;
	Mesh* mesh;
	BVHInstance bvhInstance[256];
	TLAS tlas;
	float3 p0, p1, p2; // virtual screen plane corners
	float3* accumulator;
	float* skyPixels;
	int skyWidth, skyHeight, skyBpp;
	__m256 avx_rays = _mm256_set_ps(3.0f, 2.0f, 1.0f, 0.0f, 3.0f, 2.0f, 1.0f, 0.0f);
	__m256 pixel_y = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	__m256 pixel_y2 = _mm256_set_ps(2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f);
	__m256i skywidth8, skyfull, skymul;
	__m256 skywidthpi8, skyheightpi8, scalesky;
	__m256 half8 = _mm256_set1_ps(0.5f);
	float3 lightPos = float3( 3, 10, 2 );
	__m256 lightPosx = _mm256_set1_ps(3);
	__m256 lightPosy = _mm256_set1_ps(10);
	__m256 lightPosz = _mm256_set1_ps(2);
	float3 lightColor = float3( 150, 150, 120 );
	__m256 lightColorx = _mm256_set1_ps(150);
	__m256 lightColory = _mm256_set1_ps(150);
	__m256 lightColorz = _mm256_set1_ps(120);
	float3 ambient = float3( 0.2f, 0.2f, 0.4f );
	__m256 ambientx = _mm256_set1_ps(0.2f);
	__m256 ambienty = _mm256_set1_ps(0.2f);
	__m256 ambientz = _mm256_set1_ps(0.4f);
	__m256 invscrwidth = _mm256_set1_ps(INVSCRWIDTH);
	__m256 invscrheight = _mm256_set1_ps(INVSCRHEIGHT);
	__m256 scrwidth = _mm256_set1_ps(SCRWIDTH);
	__m256 nohit = _mm256_set1_ps(1e30f); // 1e30f denotes 'no hit'

};

} // namespace Tmpl8