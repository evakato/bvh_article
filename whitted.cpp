#include "precomp.h"
#include "bvh.h"
#include "whitted.h"
#include "simdmath.h"

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 8: Whitted.
// This version shows how to build a simple Whitted-style ray tracer
// as a test case for the BVH code of the previous articles. This is
// also the final preparation for the GPGPU code in article 9.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new WhittedApp(); }

inline float3 RGB8toRGB32F( uint c )
{
	float s = 1 / 256.0f;
	int r = (c >> 16) & 255;
	int g = (c >> 8) & 255;
	int b = c & 255;
	return float3( r * s, g * s, b * s );
}

// WhittedApp implementation

void WhittedApp::Init()
{
	mesh = new Mesh( "assets/teapot.obj", "assets/bricks.png" );
	for (int i = 0; i < 16; i++)
		bvhInstance[i] = BVHInstance( mesh->bvh, i );
	tlas = TLAS( bvhInstance, 16 );
	// create a floating point accumulator for the screen
	accumulator = new float3[SCRWIDTH * SCRHEIGHT];
	// load HDR sky
	int bpp = 0;
	skyPixels = stbi_loadf( "assets/sky_19.hdr", &skyWidth, &skyHeight, &skyBpp, 0 );
	for (int i = 0; i < skyWidth * skyHeight * 3; i++) skyPixels[i] = sqrtf( skyPixels[i] );
	skyFull = skyWidth * skyHeight;
}

void WhittedApp::AnimateScene()
{
	// animate the scene
	static float a[16] = { 0 }, h[16] = { 5, 4, 3, 2, 1, 5, 4, 3 }, s[16] = { 0 };
	for (int i = 0, x = 0; x < 4; x++) for (int y = 0; y < 4; y++, i++)
	{
		mat4 R, T = mat4::Translate( (x - 1.5f) * 2.5f, 0, (y - 1.5f) * 2.5f );
		if ((x + y) & 1) R = mat4::RotateY( a[i] );
		else R = mat4::Translate( 0, h[i / 2], 0 );
		if ((a[i] += (((i * 13) & 7) + 2) * 0.005f) > 2 * PI) a[i] -= 2 * PI;
		if ((s[i] -= 0.01f, h[i] += s[i]) < 0) s[i] = 0.2f;
		bvhInstance[i].SetTransform( T * R * mat4::Scale( 1.5f ) );
	}
	// update the TLAS
	tlas.BuildQuick();
}

float3 WhittedApp::Trace( Ray& ray, int rayDepth )
{
	tlas.Intersect( ray );
	Intersection i = ray.hit;
	if (i.t == 1e30f)
	{
		// sample sky
		uint u = (uint)(skyWidth * atan2f( ray.D.z, ray.D.x ) * INV2PI - 0.5f);
		uint v = (uint)(skyHeight * acosf( ray.D.y ) * INVPI - 0.5f);
		uint skyIdx = (u + v * skyWidth) % (skyWidth * skyHeight);
		return 0.65f * float3( skyPixels[skyIdx * 3], skyPixels[skyIdx * 3 + 1], skyPixels[skyIdx * 3 + 2] );
	}
	// calculate texture uv based on barycentrics
	uint triIdx = i.instPrim & 0xfffff;
	uint instIdx = i.instPrim >> 20;
	TriEx& tri = mesh->triEx[triIdx];
	Surface* tex = mesh->texture;
	float2 uv = i.u * tri.uv1 + i.v * tri.uv2 + (1 - (i.u + i.v)) * tri.uv0;
	int iu = (int)(uv.x * tex->width) % tex->width;
	int iv = (int)(uv.y * tex->height) % tex->height;
	uint texel = tex->pixels[iu + iv * tex->width];
	float3 albedo = RGB8toRGB32F( texel );
	// calculate the normal for the intersection
	float3 N = i.u * tri.N1 + i.v * tri.N2 + (1 - (i.u + i.v)) * tri.N0;
	N = normalize( TransformVector( N, bvhInstance[instIdx].GetTransform() ) );
	float3 I = ray.O + i.t * ray.D;
	// shading
	bool mirror = (instIdx * 17) & 1;
	if (mirror)
	{	
		// calculate the specular reflection in the intersection point
		Ray secondary;
		secondary.D = ray.D - 2 * N * dot( N, ray.D );
		secondary.O = I + secondary.D * 0.001f;
		secondary.hit.t = 1e30f;
		if (rayDepth >= 10) return float3( 0 );
		return Trace( secondary, rayDepth + 1 );
	}
	else
	{
		// calculate the diffuse reflection in the intersection point
		float3 lightPos( 3, 10, 2 );
		float3 lightColor( 150, 150, 120 );
		float3 ambient( 0.2f, 0.2f, 0.4f );
		float3 L = lightPos - I;
		float dist = length( L );
		L *= 1.0f / dist;
		return albedo * (ambient + max( 0.0f, dot( N, L ) ) * lightColor * (1.0f / (dist * dist)));
	}
}

void WhittedApp::TraceAVX( RayAVX& ray, __m256* r, __m256* g, __m256* b, int rayDepth )
{
	tlas.IntersectAVX( ray );
	//print_m256(ray.t8);
	__m256 mask = _mm256_cmp_ps(ray.t8, _mm256_set1_ps(1e29f), _CMP_GT_OQ);
	__m256 half = _mm256_set1_ps(0.3f);

	// sample sky
	__m256 skyWidth8 = _mm256_set1_ps(skyWidth);
	//uint u = (uint)(skyWidth * atan2f( ray.D.z, ray.D.x ) * INV2PI - 0.5f);
	__m256 sky_tan = _mm256_mul_ps(skyWidth8, atan2_avx(ray.Dz8, ray.Dx8));
	__m256 sky_u8 = _mm256_fmsub_ps(sky_tan, _mm256_set1_ps(INV2PI), _mm256_set1_ps(0.5f));

	__m256i int_vector = _mm256_cvtps_epu32(sky_u8);
	unsigned int x[8];
	_mm256_storeu_si256((__m256i*)x, int_vector);
	printf("%d \n", x[0]);

	//uint v = (uint)(skyHeight * acosf( ray.D.y ) * INVPI - 0.5f);
	__m256 sky_cos = _mm256_mul_ps(_mm256_set1_ps(skyHeight), _mm256_acos_ps(ray.Dy8));
	__m256 sky_v8 = _mm256_fmsub_ps(sky_cos, _mm256_set1_ps(INV2PI), _mm256_set1_ps(0.5f));

	//uint skyIdx = (u + v * skyWidth) % skyFull;
	__m256 skyIdx = modAVX(_mm256_fmadd_ps(sky_v8, skyWidth8, sky_u8), _mm256_set1_ps(skyFull));

	//return 0.65f * float3( skyPixels[skyIdx * 3], skyPixels[skyIdx * 3 + 1], skyPixels[skyIdx * 3 + 2] );
	__m256i intIndices = _mm256_cvttps_epi32(skyIdx);
	int skyIdxI[8];
	_mm256_storeu_si256((__m256i*)skyIdxI, intIndices);
	int skyrI[8], skygI[8], skybI[8];
	for (int i = 0; i < 8; i++) {
		int scaleI = skyIdxI[i] * 3;
		skyrI[i] = skyPixels[scaleI];
		skygI[i] = skyPixels[scaleI + 1];
		skybI[i] = skyPixels[scaleI + 2];
	}
	__m256i skyrI8 = _mm256_loadu_si256((__m256i*)skyrI);
	__m256 skyr = _mm256_cvtepi32_ps(skyrI8);
	__m256i skygI8 = _mm256_loadu_si256((__m256i*)skygI);
	__m256 skyg = _mm256_cvtepi32_ps(skygI8);
	__m256i skybI8 = _mm256_loadu_si256((__m256i*)skybI);
	__m256 skyb = _mm256_cvtepi32_ps(skybI8);

	__m256 scale = _mm256_set1_ps(0.65f);
	//skyr = _mm256_mul_ps(skyr, scale);
	//skyg = _mm256_mul_ps(skyg, scale);
	//skyg = _mm256_mul_ps(skyb, scale);
	skyr = _mm256_set1_ps(0.5f);
	skyg = _mm256_set1_ps(0.5f);
	skyb = _mm256_set1_ps(0.5f);

	*r = _mm256_blendv_ps(half, skyr, mask);
	*g = _mm256_blendv_ps(half, skyg, mask);
	*b = _mm256_blendv_ps(half, skyb, mask);

	/*
	// calculate texture uv based on barycentrics
	uint triIdx = i.instPrim & 0xfffff;
	uint instIdx = i.instPrim >> 20;
	TriEx& tri = mesh->triEx[triIdx];
	Surface* tex = mesh->texture;
	float2 uv = i.u * tri.uv1 + i.v * tri.uv2 + (1 - (i.u + i.v)) * tri.uv0;
	int iu = (int)(uv.x * tex->width) % tex->width;
	int iv = (int)(uv.y * tex->height) % tex->height;
	uint texel = tex->pixels[iu + iv * tex->width];
	float3 albedo = RGB8toRGB32F( texel );
	// calculate the normal for the intersection
	float3 N = i.u * tri.N1 + i.v * tri.N2 + (1 - (i.u + i.v)) * tri.N0;
	N = normalize( TransformVector( N, bvhInstance[instIdx].GetTransform() ) );
	float3 I = ray.O + i.t * ray.D;
	// shading
	bool mirror = (instIdx * 17) & 1;
	if (mirror)
	{	
		// calculate the specular reflection in the intersection point
		Ray secondary;
		secondary.D = ray.D - 2 * N * dot( N, ray.D );
		secondary.O = I + secondary.D * 0.001f;
		secondary.hit.t = 1e30f;
		if (rayDepth >= 10) return float3( 0 );
		return Trace( secondary, rayDepth + 1 );
	}
	else
	{
		// calculate the diffuse reflection in the intersection point
		float3 lightPos( 3, 10, 2 );
		float3 lightColor( 150, 150, 120 );
		float3 ambient( 0.2f, 0.2f, 0.4f );
		float3 L = lightPos - I;
		float dist = length( L );
		L *= 1.0f / dist;
		return albedo * (ambient + max( 0.0f, dot( N, L ) ) * lightColor * (1.0f / (dist * dist)));
	}
	*/
}
void WhittedApp::Tick( float deltaTime )
{
	Timer t;
	// update the TLAS
	AnimateScene();
	// render the scene: multithreaded tiles
	static float angle = 0; angle += 0.01f;
	mat4 M1 = mat4::RotateY( angle ), M2 = M1 * mat4::RotateX( -0.65f );
	// setup screen plane in world space
	float aspectRatio = (float)SCRWIDTH / SCRHEIGHT;
	p0 = TransformPosition( float3( -aspectRatio, 1, 1.5f ), M2 );
	p1 = TransformPosition( float3( aspectRatio, 1, 1.5f ), M2 );
	p2 = TransformPosition( float3( -aspectRatio, -1, 1.5f ), M2 );
	float3 camPos = TransformPosition( float3( 0, -2, -8.5f ), M1 );
#pragma omp parallel for schedule(dynamic)
	for (int tile = 0; tile < (SCRWIDTH * SCRHEIGHT / 16); tile++)
	{
		// render an 4x4 tile
		int x = tile % (SCRWIDTH / 4), y = tile / (SCRWIDTH / 4);
#if 0
		Ray ray;
		ray.O = camPos;
		for (int v = 0; v < 4; v++) {
			for (int u = 0; u < 4; u++) {
				// setup a primary ray
				//union { float3 pixelPos; __m128 pixelPos4; };
				float3 pixelPos = ray.O + p0 +
					(p1 - p0) * ((x * 4 + u + RandomFloat()) / SCRWIDTH) +
					(p2 - p0) * ((y * 4 + v + RandomFloat()) / SCRHEIGHT);
				ray.D = normalize(pixelPos - ray.O);
				//ray.D4 = normalizeSIMD( _mm_sub_ps(pixelPos4, ray.O4) );
				ray.hit.t = 1e30f; // 1e30f denotes 'no hit'
				uint pixelAddress = x * 4 + u + (y * 4 + v) * SCRWIDTH;
				accumulator[pixelAddress] = Trace(ray);
			}
		}
#else
		RayAVX ray;
		ray.Ox8 = _mm256_set1_ps(camPos.x);
		ray.Oy8 = _mm256_set1_ps(camPos.y);
		ray.Oz8 = _mm256_set1_ps(camPos.z);
		for (int halfsquare = 0; halfsquare < 4; halfsquare+=2) {
			float3 pixelPos0 = p0 +
				(p1 - p0) * ((x * 4 + 0 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 0 + halfsquare + RandomFloat()) / SCRHEIGHT);
			float3 pixelPos1 = p0 +
				(p1 - p0) * ((x * 4 + 1 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 0 + halfsquare + RandomFloat()) / SCRHEIGHT);
			float3 pixelPos2 = p0 +
				(p1 - p0) * ((x * 4 + 2 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 0 + halfsquare + RandomFloat()) / SCRHEIGHT);
			float3 pixelPos3 = p0 +
				(p1 - p0) * ((x * 4 + 3 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 0 + halfsquare + RandomFloat()) / SCRHEIGHT);
			float3 pixelPos4 = p0 +
				(p1 - p0) * ((x * 4 + 0 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 1 + halfsquare + RandomFloat()) / SCRHEIGHT);
			float3 pixelPos5 = p0 +
				(p1 - p0) * ((x * 4 + 1 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 1 + halfsquare + RandomFloat()) / SCRHEIGHT);
			float3 pixelPos6 = p0 +
				(p1 - p0) * ((x * 4 + 2 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 1 + halfsquare + RandomFloat()) / SCRHEIGHT);
			float3 pixelPos7 = p0 +
				(p1 - p0) * ((x * 4 + 3 + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 4 + 1 + halfsquare + RandomFloat()) / SCRHEIGHT);
			__m256 pixelPosx = _mm256_set_ps(pixelPos7.x, pixelPos6.x, pixelPos5.x, pixelPos4.x, pixelPos3.x, pixelPos2.x, pixelPos1.x, pixelPos0.x);
			__m256 pixelPosy = _mm256_set_ps(pixelPos7.y, pixelPos6.y, pixelPos5.y, pixelPos4.y, pixelPos3.y, pixelPos2.y, pixelPos1.y, pixelPos0.y);
			__m256 pixelPosz = _mm256_set_ps(pixelPos7.z, pixelPos6.z, pixelPos5.z, pixelPos4.z, pixelPos3.z, pixelPos2.z, pixelPos1.z, pixelPos0.z);
			//pixelPosx = _mm256_add_ps(pixelPosx, ray.Ox8);
			//pixelPosy = _mm256_add_ps(pixelPosy, ray.Oy8);
			//pixelPosz = _mm256_add_ps(pixelPosz, ray.Oz8);
			//ray.D = normalize(pixelPos - ray.O);
			normalizeAVX(&pixelPosx, &pixelPosy, &pixelPosz);
			ray.Dx8 = pixelPosx;
			ray.Dy8 = pixelPosy;
			ray.Dz8 = pixelPosz;

			ray.t8 = _mm256_set1_ps(1e30f); // 1e30f denotes 'no hit'

			__m256 r, g, b;
			TraceAVX(ray, &r, &g, &b);
			float r_out[8], g_out[8], b_out[8];
			_mm256_storeu_ps(r_out, r);
			_mm256_storeu_ps(g_out, g);
			_mm256_storeu_ps(b_out, b);

			int i = 0;
			for (int v = 0; v < 2; v++) {
				for (int u = 0; u < 4; u++) {
					uint pixelAddress = x * 4 + u + (y * 4 + v + halfsquare) * SCRWIDTH;
					accumulator[pixelAddress].x = r_out[i];
					accumulator[pixelAddress].y = g_out[i];
					accumulator[pixelAddress].z = b_out[i];
					i++;
				}
			}
		}
#endif
	}
	// convert the floating point accumulator into pixels
	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++)
	{
		int r = min( 255, (int)(255 * accumulator[i].x) );
		int g = min( 255, (int)(255 * accumulator[i].y) );
		int b = min( 255, (int)(255 * accumulator[i].z) );
		screen->pixels[i] = (r << 16) + (g << 8) + b;
	}
	// performance report - running average - ms, MRays/s
	static float avg = 10, alpha = 1;
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.5f;
	float fps = 1000.0f / avg, rps = (SCRWIDTH * SCRHEIGHT) / avg;
	printf("%5.2fms (%.1ffps) - %.1fMrays/s\n", avg, fps, rps / 1000);
}

// EOF