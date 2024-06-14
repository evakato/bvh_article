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

__m256 avx_rays = _mm256_set_ps(3.0f, 2.0f, 1.0f, 0.0f, 3.0f, 2.0f, 1.0f, 0.0f);
float3 lightPos( 3, 10, 2 );
float3 lightColor( 150, 150, 120 );
float3 ambient( 0.2f, 0.2f, 0.4f );

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
	if (false && mirror)
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

//void WhittedApp::TraceAVX( RayAVX& ray, __m256* r, __m256* g, __m256* b, int rayDepth )
void WhittedApp::TraceAVX( RayAVX& ray, float3* albedo, int rayDepth )
{
	tlas.IntersectAVX( ray );

	float Dx[8], Dy[8], Dz[8];
	float rayt[8];
	float Ox[8], Oy[8], Oz[8];
	int triIdxs[8], instIdxs[8];
	_mm256_storeu_ps(Dx, ray.Dx8);
	_mm256_storeu_ps(Dy, ray.Dy8);
	_mm256_storeu_ps(Dz, ray.Dz8);
	_mm256_storeu_ps(rayt, ray.t8);
	_mm256_storeu_ps(Ox, ray.Ox8);
	_mm256_storeu_ps(Oy, ray.Oy8);
	_mm256_storeu_ps(Oz, ray.Oz8);

	Surface* tex = mesh->texture;

	__m256i triIdx = _mm256_cvtps_epi32(ray.triIdx);
	__m256i instIdx = _mm256_cvtps_epi32(ray.instIdx);
	_mm256_storeu_si256((__m256i*)triIdxs, triIdx);
	_mm256_storeu_si256((__m256i*)instIdxs, instIdx);

	for (int i = 0; i < 8; i++) {
		if (rayt[i] >= 1e30f) {
			uint u = (uint)(skyWidth * atan2f(Dz[i], Dx[i]) * INV2PI - 0.5f);
			uint v = (uint)(skyHeight * acosf(Dy[i]) * INVPI - 0.5f);
			uint skyIdx = (u + v * skyWidth) % (skyFull);
			albedo[i] = 0.65f * float3(skyPixels[skyIdx * 3], skyPixels[skyIdx * 3 + 1], skyPixels[skyIdx * 3 + 2]);
		}
		else {
			TriEx& tri = mesh->triEx[triIdxs[i]];
			float rayu = ray.u.m256_f32[i];
			float rayv = ray.v.m256_f32[i];
			float rayw = rayu + rayv;
			float2 uv = rayu * tri.uv1 + rayv * tri.uv2 + (1 - rayw) * tri.uv0;
			int iu = (int)(uv.x * tex->width) % tex->width;
			int iv = (int)(uv.y * tex->height) % tex->height;
			uint texel = tex->pixels[iu + iv * tex->width];
			float3 tex_albedo = RGB8toRGB32F(texel);

			float3 N = rayu * tri.N1 + rayv * tri.N2 + (1 - rayw) * tri.N0;
			N = normalize( TransformVector( N, bvhInstance[instIdxs[i]].GetTransform() ) );
			float3 O = float3(Ox[i], Oy[i], Oz[i]);
			float3 D = float3(Dx[i], Dy[i], Dz[i]);
			float3 I = O + rayt[i] * D;

			bool mirror = (instIdxs[i] * 17) & 1;
			if (false && mirror)
			{	
				// calculate the specular reflection in the intersection point
				Ray secondary;
				secondary.D = D - 2 * N * dot( N, D );
				secondary.O = I + secondary.D * 0.001f;
				secondary.hit.t = 1e30f;
				if (rayDepth >= 10) albedo[i] = float3(0);
				albedo[i] = Trace( secondary, rayDepth + 1 );
			}
			float3 L = lightPos - I;
			float dist = length( L );
			L *= 1.0f / dist;
			albedo[i] = tex_albedo * (ambient + max( 0.0f, dot( N, L ) ) * lightColor * (1.0f / (dist * dist)));
		}
	}
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

		float3 pwidth = p1 - p0;
		float3 pheight = p2 - p0;

		int scalex = x * 4;
		int scaley = y * 4 + 1;
		int scaley2 = scaley + 2;

		// Precalculate common terms
		__m256 invscrwidth = _mm256_set1_ps(INVSCRWIDTH);
		__m256 invscrheight = _mm256_set1_ps(INVSCRHEIGHT);
		__m256 p0x = _mm256_set1_ps(p0.x);
		__m256 p0y = _mm256_set1_ps(p0.y);
		__m256 p0z = _mm256_set1_ps(p0.z);
		__m256 pwidth_x = _mm256_set1_ps(pwidth.x);
		__m256 pwidth_y = _mm256_set1_ps(pwidth.y);
		__m256 pwidth_z = _mm256_set1_ps(pwidth.z);
		__m256 pheight_x = _mm256_set1_ps(pheight.x);
		__m256 pheight_y = _mm256_set1_ps(pheight.y);
		__m256 pheight_z = _mm256_set1_ps(pheight.z);
		__m256 avx_rays_plus_scalex = _mm256_add_ps(_mm256_set1_ps(scalex), avx_rays);
		__m256 random_floats1 = generate_random_floats();
		__m256 random_floats2 = generate_random_floats();

		// Loop unroll 1
		__m256 pixelPosx = _mm256_add_ps(_mm256_add_ps(p0x, _mm256_mul_ps(pwidth_x, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth))), _mm256_mul_ps(pheight_x, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley), random_floats2), invscrheight)));
		__m256 pixelPosy = _mm256_add_ps(_mm256_add_ps(p0y, _mm256_mul_ps(pwidth_y, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth))), _mm256_mul_ps(pheight_y, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley), random_floats2), invscrheight)));
		__m256 pixelPosz = _mm256_add_ps(_mm256_add_ps(p0z, _mm256_mul_ps(pwidth_z, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth))), _mm256_mul_ps(pheight_z, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley), random_floats2), invscrheight)));

		normalizeAVX(&pixelPosx, &pixelPosy, &pixelPosz);
		ray.Dx8 = pixelPosx;
		ray.Dy8 = pixelPosy;
		ray.Dz8 = pixelPosz;
		ray.t8 = _mm256_set1_ps(1e30f); // 1e30f denotes 'no hit'

		__m256 r, g, b;
		float3 albedo[8];
		TraceAVX(ray, albedo);

		float r_out[8], g_out[8], b_out[8];
		_mm256_storeu_ps(r_out, r);
		_mm256_storeu_ps(g_out, g);
		_mm256_storeu_ps(b_out, b);

		int i = 0;
		for (int v = 0; v < 2; v++) {
			for (int u = 0; u < 4; u++) {
				uint pixelAddress = x * 4 + u + (y * 4 + v) * SCRWIDTH;
				accumulator[pixelAddress] = albedo[i];
				//accumulator[pixelAddress].x = r_out[i];
				//accumulator[pixelAddress].y = g_out[i];
				//accumulator[pixelAddress].z = b_out[i];
				i++;
			}
		}

		// Loop unroll 2
		random_floats1 = generate_random_floats();
		random_floats2 = generate_random_floats();

		pixelPosx = _mm256_add_ps(_mm256_add_ps(p0x, _mm256_mul_ps(pwidth_x, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth))), _mm256_mul_ps(pheight_x, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley2), random_floats2), invscrheight)));
		pixelPosy = _mm256_add_ps(_mm256_add_ps(p0y, _mm256_mul_ps(pwidth_y, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth))), _mm256_mul_ps(pheight_y, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley2), random_floats2), invscrheight)));
		pixelPosz = _mm256_add_ps(_mm256_add_ps(p0z, _mm256_mul_ps(pwidth_z, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth))), _mm256_mul_ps(pheight_z, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley2), random_floats2), invscrheight)));

		normalizeAVX(&pixelPosx, &pixelPosy, &pixelPosz);
		ray.Dx8 = pixelPosx;
		ray.Dy8 = pixelPosy;
		ray.Dz8 = pixelPosz;
		ray.t8 = _mm256_set1_ps(1e30f); // 1e30f denotes 'no hit'

		TraceAVX(ray, albedo);
		_mm256_storeu_ps(r_out, r);
		_mm256_storeu_ps(g_out, g);
		_mm256_storeu_ps(b_out, b);

		i = 0;
		for (int v = 0; v < 2; v++) {
			for (int u = 0; u < 4; u++) {
				uint pixelAddress = x * 4 + u + (y * 4 + v + 2) * SCRWIDTH;
				accumulator[pixelAddress] = albedo[i];
				//accumulator[pixelAddress].x = r_out[i];
				//accumulator[pixelAddress].y = g_out[i];
				//accumulator[pixelAddress].z = b_out[i];
				i++;
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