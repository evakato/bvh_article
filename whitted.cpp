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
const __m256i maskR = _mm256_set1_epi32(0x00FF0000);
const __m256i maskG = _mm256_set1_epi32(0x0000FF00);
const __m256i maskB = _mm256_set1_epi32(0x000000FF);
const __m256 scale = _mm256_set1_ps(1.0f / 256.0f);

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
	// hardcoded this
	skyfull = _mm256_set1_epi32(0xFFFFFF);
	skymul = _mm256_set1_epi32(3);
	skywidth8 = _mm256_set1_epi32(skyWidth);
	skywidthpi8 = _mm256_set1_ps(INV2PI * skyWidth);
	skyheightpi8 = _mm256_set1_ps(INVPI * skyHeight);
	scalesky = _mm256_set1_ps(0.65f);
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
		float3 L = lightPos - I;
		float dist = length( L );
		L *= 1.0f / dist;
		return albedo * (ambient + max( 0.0f, dot( N, L ) ) * lightColor * (1.0f / (dist * dist)));
	}
}

void WhittedApp::TraceAVX( RayAVX& ray, __m256* r, __m256* g, __m256* b, int rayDepth )
//void WhittedApp::TraceAVX( RayAVX& ray, float3* albedo, int rayDepth )
{
	tlas.IntersectAVX( ray );

	// does the ray hit skydome
	__m256 hitsSky = _mm256_cmp_ps(ray.t8, _mm256_set1_ps(1e30f), _CMP_GE_OQ);
	// get sky albedos
	__m256 skyu = _mm256_sub_ps(_mm256_mul_ps(_mm256_atan2_ps(ray.Dz8, ray.Dx8), skywidthpi8), half8);
	__m256 skyv = _mm256_sub_ps(_mm256_mul_ps(_mm256_acos_ps(ray.Dy8), skyheightpi8), half8);
	__m256i skyui = _mm256_cvtps_epi32(skyu);
	__m256i skyvi = _mm256_cvtps_epi32(skyv);
	__m256i skyIdx8 = (_mm256_add_epi32(_mm256_mullo_epi32(skyvi, skywidth8), skyui));
	skyIdx8 = _mm256_mullo_epi32(_mm256_and_si256(skyIdx8, skyfull), skymul);
	__m256 skyx = _mm256_mul_ps(_mm256_i32gather_ps(skyPixels, skyIdx8, sizeof(float)), scalesky);
	__m256 skyy = _mm256_mul_ps(_mm256_i32gather_ps(skyPixels, _mm256_add_epi32(skyIdx8, _mm256_set1_epi32(1)), sizeof(float)), scalesky);
	__m256 skyz = _mm256_mul_ps(_mm256_i32gather_ps(skyPixels, _mm256_add_epi32(skyIdx8, _mm256_set1_epi32(2)), sizeof(float)), scalesky);
	*r = _mm256_blendv_ps(_mm256_setzero_ps(), skyx, hitsSky);
	*g = _mm256_blendv_ps(_mm256_setzero_ps(), skyy, hitsSky);
	*b = _mm256_blendv_ps(_mm256_setzero_ps(), skyz, hitsSky);
	int skyMask = _mm256_movemask_ps(hitsSky);
	if (skyMask == 255) return;

	int instTriIdxs[8];
	__m256i instTriIdx8 = _mm256_cvtps_epi32(ray.instTriIdx);
	_mm256_storeu_si256((__m256i*)instTriIdxs, instTriIdx8);
	Surface* tex = mesh->texture;

	__m256 rayw = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(ray.u, ray.v));
	__m256i triIdx = _mm256_and_si256(instTriIdx8, _mm256_set1_epi32(0xFFFFF));
	__m256i instIdx = _mm256_srli_epi32(instTriIdx8, 20);
	__m256 Ix = _mm256_add_ps(ray.Ox8, _mm256_mul_ps(ray.t8, ray.Dx8));
	__m256 Iy = _mm256_add_ps(ray.Oy8, _mm256_mul_ps(ray.t8, ray.Dy8));
	__m256 Iz = _mm256_add_ps(ray.Oz8, _mm256_mul_ps(ray.t8, ray.Dz8));

	for (int i = 0; i < 8; ++i) {
		if ((skyMask & (1 << i)) == 0) {
			__m256 matchesRay = _mm256_cmp_ps(_mm256_set1_ps(ray.instTriIdx.m256_f32[i]), ray.instTriIdx, _CMP_EQ_OQ);
			TriEx& tri = mesh->triEx[triIdx.m256i_i32[i]];

			// get texture
			__m256 uvx = _mm256_fmadd_ps(ray.v, _mm256_set1_ps(tri.uv2.x), _mm256_fmadd_ps(ray.u, _mm256_set1_ps(tri.uv1.x), _mm256_mul_ps(rayw, _mm256_set1_ps(tri.uv0.x))));
			__m256 uvy = _mm256_fmadd_ps(ray.v, _mm256_set1_ps(tri.uv2.y), _mm256_fmadd_ps(ray.u, _mm256_set1_ps(tri.uv1.y), _mm256_mul_ps(rayw, _mm256_set1_ps(tri.uv0.y))));
			__m256i iu = _mm256_and_si256(_mm256_cvttps_epi32(_mm256_mul_ps(uvx, _mm256_set1_ps(1024.0f))), _mm256_set1_epi32(1023));
			__m256i iv = _mm256_and_si256(_mm256_cvttps_epi32(_mm256_mul_ps(uvy, _mm256_set1_ps(1024.0f))), _mm256_set1_epi32(1023));
			__m256i texel_indices = _mm256_add_epi32(iu, _mm256_mullo_epi32(iv, _mm256_set1_epi32(1024)));
			__m256i texels = _mm256_i32gather_epi32((const int*)tex->pixels, texel_indices, 4);
			__m256i redInt = _mm256_srli_epi32(_mm256_and_si256(texels, maskR), 16);
			__m256i greenInt = _mm256_srli_epi32(_mm256_and_si256(texels, maskG), 8);
			__m256i blueInt = _mm256_and_si256(texels, maskB);
			__m256 red = _mm256_cvtepi32_ps(redInt);
			__m256 green = _mm256_cvtepi32_ps(greenInt);
			__m256 blue = _mm256_cvtepi32_ps(blueInt);
			__m256 Nx = _mm256_fmadd_ps(ray.v, _mm256_set1_ps(tri.N2.x), _mm256_fmadd_ps(ray.u, _mm256_set1_ps(tri.N1.x), _mm256_mul_ps(rayw, _mm256_set1_ps(tri.N0.x))));
			__m256 Ny = _mm256_fmadd_ps(ray.v, _mm256_set1_ps(tri.N2.y), _mm256_fmadd_ps(ray.u, _mm256_set1_ps(tri.N1.y), _mm256_mul_ps(rayw, _mm256_set1_ps(tri.N0.y))));
			__m256 Nz = _mm256_fmadd_ps(ray.v, _mm256_set1_ps(tri.N2.z), _mm256_fmadd_ps(ray.u, _mm256_set1_ps(tri.N1.z), _mm256_mul_ps(rayw, _mm256_set1_ps(tri.N0.z))));

			TransformVectorAVX(Nx, Ny, Nz, bvhInstance[instIdx.m256i_i32[i]].GetTransform());
			normalizeAVX(&Nx, &Ny, &Nz);

			__m256 Lx = _mm256_sub_ps(lightPosx, Ix), Ly = _mm256_sub_ps(lightPosy, Iy), Lz = _mm256_sub_ps(lightPosz, Iz);
			__m256 dist = avxLength(Lx, Ly, Lz);
			__m256 invDist = _mm256_rcp_ps(dist);
			Lx = _mm256_mul_ps(Lx, invDist), Ly = _mm256_mul_ps(Ly, invDist), Lz = _mm256_mul_ps(Lz, invDist);
			__m256 NdotL = dotAVX(Nx, Ny, Nz, Lx, Ly, Lz);
			NdotL = _mm256_max_ps(_mm256_set1_ps(0.0f), NdotL);
			__m256 distAtten = _mm256_mul_ps(invDist, invDist);
			__m256 albedor = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(red, _mm256_add_ps(ambientx, NdotL)), lightColorx), distAtten);
			__m256 albedog = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(green, _mm256_add_ps(ambienty, NdotL)), lightColory), distAtten);
			__m256 albedob = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(blue, _mm256_add_ps(ambientz, NdotL)), lightColorz), distAtten);

			*r = _mm256_blendv_ps(*r, _mm256_mul_ps(albedor, scale), matchesRay);
			*g = _mm256_blendv_ps(*g, _mm256_mul_ps(albedog, scale), matchesRay);
			*b = _mm256_blendv_ps(*b, _mm256_mul_ps(albedob, scale), matchesRay);
			skyMask = _mm256_movemask_ps(_mm256_or_ps(matchesRay, hitsSky));
			if (skyMask == 255) return;
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

		// Precalculate common terms
		float3 pwidth = p1 - p0, pheight = p2 - p0;
		int scalex = x * 4;
		int scaleym = y * 4;
		int scaley = scaleym + 1;
		int scaley2 = scaley + 2;
		__m256 p0x = _mm256_set1_ps(p0.x), p0y = _mm256_set1_ps(p0.y), p0z = _mm256_set1_ps(p0.z);
		__m256 pwidth_x = _mm256_set1_ps(pwidth.x), pwidth_y = _mm256_set1_ps(pwidth.y), pwidth_z = _mm256_set1_ps(pwidth.z);
		__m256 pheight_x = _mm256_set1_ps(pheight.x), pheight_y = _mm256_set1_ps(pheight.y), pheight_z = _mm256_set1_ps(pheight.z);
		__m256 avx_rays_plus_scalex = _mm256_add_ps(_mm256_set1_ps(scalex), avx_rays);
		__m256 pixel_y_plus_scaley = _mm256_add_ps(_mm256_set1_ps(scaleym), pixel_y);
		__m256 pixel_y2_plus_scaley = _mm256_add_ps(_mm256_set1_ps(scaleym), pixel_y2);
		__m256 scaley_vec = _mm256_set1_ps(scaley);
		__m256 scaley2_vec = _mm256_set1_ps(scaley2);
		__m256 random_floats1 = generate_random_floats();
		__m256 random_floats2 = generate_random_floats();

		// Loop unroll 1
		__m256 pixelPosx = _mm256_fmadd_ps(pwidth_x, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth), p0x);
		pixelPosx = _mm256_fmadd_ps(pheight_x, _mm256_mul_ps(_mm256_add_ps(scaley_vec, random_floats2), invscrheight), pixelPosx);
		__m256 pixelPosy = _mm256_fmadd_ps(pwidth_y, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth), p0y);
		pixelPosy = _mm256_fmadd_ps(pheight_y, _mm256_mul_ps(_mm256_add_ps(scaley_vec, random_floats2), invscrheight), pixelPosy);
		__m256 pixelPosz = _mm256_fmadd_ps(pwidth_z, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth), p0z);
		pixelPosz = _mm256_fmadd_ps(pheight_z, _mm256_mul_ps(_mm256_add_ps(scaley_vec, random_floats2), invscrheight), pixelPosz);

		normalizeAVX(&pixelPosx, &pixelPosy, &pixelPosz);
		ray.Dx8 = pixelPosx;
		ray.Dy8 = pixelPosy;
		ray.Dz8 = pixelPosz;
		ray.t8 = nohit;

		__m256 r, g, b;
		TraceAVX(ray, &r, &g, &b);

		__m256i pixelAddy = _mm256_cvttps_epi32(_mm256_fmadd_ps(scrwidth, pixel_y_plus_scaley, avx_rays_plus_scalex));
		float r_out[8], g_out[8], b_out[8];
		uint pixel_a[8];
		_mm256_storeu_ps(r_out, r);
		_mm256_storeu_ps(g_out, g);
		_mm256_storeu_ps(b_out, b);
		_mm256_storeu_si256((__m256i*)pixel_a, pixelAddy);

		for (int k = 0; k < 8; k++) {
			accumulator[pixel_a[k]].x = r_out[k];
			accumulator[pixel_a[k]].y = g_out[k];
			accumulator[pixel_a[k]].z = b_out[k];
		}

		// Loop unroll 2
		pixelPosx = _mm256_fmadd_ps(pheight_x, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley2), random_floats2), invscrheight), _mm256_fmadd_ps(pwidth_x, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth), p0x));
		pixelPosy = _mm256_fmadd_ps(pheight_y, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley2), random_floats2), invscrheight), _mm256_fmadd_ps(pwidth_y, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth), p0y));
		pixelPosz = _mm256_fmadd_ps(pheight_z, _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(scaley2), random_floats2), invscrheight), _mm256_fmadd_ps(pwidth_z, _mm256_mul_ps(_mm256_add_ps(avx_rays_plus_scalex, random_floats1), invscrwidth), p0z));

		normalizeAVX(&pixelPosx, &pixelPosy, &pixelPosz);
		ray.Dx8 = pixelPosx;
		ray.Dy8 = pixelPosy;
		ray.Dz8 = pixelPosz;
		ray.t8 = nohit;

		TraceAVX(ray, &r, &g, &b);
		pixelAddy = _mm256_cvttps_epi32(_mm256_fmadd_ps(scrwidth, pixel_y2_plus_scaley, avx_rays_plus_scalex));
		_mm256_storeu_ps(r_out, r);
		_mm256_storeu_ps(g_out, g);
		_mm256_storeu_ps(b_out, b);
		_mm256_storeu_si256((__m256i*)pixel_a, pixelAddy);

		for (int k = 0; k < 8; k++) {
			accumulator[pixel_a[k]].x = r_out[k];
			accumulator[pixel_a[k]].y = g_out[k];
			accumulator[pixel_a[k]].z = b_out[k];
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