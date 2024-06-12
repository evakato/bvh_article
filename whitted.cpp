#include "precomp.h"
#include "bvh.h"
#include "whitted.h"

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 8: Whitted.
// This version shows how to build a simple Whitted-style ray tracer
// as a test case for the BVH code of the previous articles. This is
// also the final preparation for the GPGPU code in article 9.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new WhittedApp(); }

/*
static union { __m128 lightColor4; float3 lightColor; };
lightColor = float3( 150, 150, 120 );
union { __m128 ambient4; float3 ambient; };
ambient = float3( 0.2f, 0.2f, 0.4f );
union { __m128 lightPos4; float3 lightPos; };
lightPos = float3( 3, 10, 2 );
*/

__m128 lightColor4 = _mm_set_ps(0.0f, 120.0f, 150.0f, 150.0f);
__m128 ambient4 = _mm_set_ps(0.0f, 0.4f, 0.2f, 0.2f);
__m128 lightPos4 = _mm_set_ps(0.0f, 2.0f, 10.0f, 3.0f);

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
	__m128 iu4 = _mm_set_ps1(i.u);
	__m128 iv4 = _mm_set_ps1(i.v);
	__m128 iw4 = _mm_sub_ps(_mm_set_ps1(1), _mm_add_ps(iu4, iv4));
	__m128 bary1 = _mm_mul_ps(iu4, tri.uv1);
	__m128 bary2 = _mm_mul_ps(iv4, tri.uv2);
	__m128 bary0 = _mm_mul_ps(iw4, tri.uv0);
	union { __m128 uv; float2 uvi; };
	uv = _mm_add_ps(bary1, _mm_add_ps(bary2, bary0));
#if 0
	int iu = (int)(uvi.x * tex->width) % tex->width;
	int iv = (int)(uvi.y * tex->height) % tex->height;
	uint texel = tex->pixels[iu + iv * tex->width];
#else 
	__m128 texSize = _mm_set_ps(0.0f, 0.0f, (float)tex->height, (float)tex->width);
	__m128 scaledUV = _mm_mul_ps(uv, texSize);
	// Convert to integers
	__m128i intUV = _mm_cvttps_epi32(scaledUV);
	// Compute modulus using bitwise AND with (dimension - 1) for power of 2 texture sizes
	__m128i texDimMask = _mm_set_epi32(0, 0, tex->height - 1, tex->width - 1);
	intUV = _mm_and_si128(intUV, texDimMask);
	// Extract the resulting integer coordinates
	int iu = _mm_extract_epi32(intUV, 0);
	int iv = _mm_extract_epi32(intUV, 1);
	uint texel = tex->pixels[iu + iv * tex->width];
#endif
	union { __m128 albedo4; float3 albedo; };
	albedo = RGB8toRGB32F( texel );
	// calculate the normal for the intersection
	//float3 N = i.u * tri.normal1 + i.v * tri.normal2 + (1 - (i.u + i.v)) * tri.normal0;
	bary1 = _mm_mul_ps(iu4, tri.n1);
	bary2 = _mm_mul_ps(iv4, tri.n2);
	bary0 = _mm_mul_ps(iw4, tri.n0);
	union { __m128 n4; float3 N; };
	n4 = _mm_add_ps(bary1, _mm_add_ps(bary2, bary0));
	N = normalize( TransformVector( N, bvhInstance[instIdx].GetTransform() ) );
	union { __m128 i4; float3 I; };
	i4 = _mm_add_ps(ray.O4, _mm_mul_ps(_mm_set_ps1(i.t), ray.D4));
	//float3 I = ray.O + i.t * ray.D;
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
		__m128 L4 = _mm_sub_ps(lightPos4, i4);
		__m128 dist4 = _mm_sqrt_ps(_mm_dp_ps(L4, L4, 0xFF));
		__m128 invdist4 = _mm_div_ps(_mm_set_ps1(1.0f), dist4);
		L4 = _mm_mul_ps(L4, invdist4);
		__m128 NdotL = _mm_max_ps(_mm_set_ps1(0.0f), _mm_dp_ps(n4, L4, 0xFF));
		__m128 distatten = _mm_div_ps(_mm_set_ps1(1.0f), _mm_mul_ps(dist4, dist4));
		__m128 shading = _mm_mul_ps(distatten, _mm_mul_ps(NdotL, lightColor4));
		union { __m128 finalcol4; float3 finalcol; };
		finalcol4 = _mm_mul_ps(albedo4, _mm_add_ps(ambient4, shading));
		return finalcol;
		//return albedo * (ambient + max( 0.0f, dot( N, L ) ) * lightColor * (1.0f / (dist * dist)));
	}
}

void WhittedApp::Tick( float deltaTime )
{
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
	for (int tile = 0; tile < (SCRWIDTH * SCRHEIGHT / 64); tile++)
	{
		// render an 8x8 tile
		int x = tile % (SCRWIDTH / 8), y = tile / (SCRWIDTH / 8);
		Ray ray;
		ray.O = camPos;
		for (int v = 0; v < 8; v++) for (int u = 0; u < 8; u++)
		{
			// setup a primary ray
			float3 pixelPos = ray.O + p0 +
				(p1 - p0) * ((x * 8 + u + RandomFloat()) / SCRWIDTH) +
				(p2 - p0) * ((y * 8 + v + RandomFloat()) / SCRHEIGHT);
			ray.D = normalize( pixelPos - ray.O );
			ray.hit.t = 1e30f; // 1e30f denotes 'no hit'
			uint pixelAddress = x * 8 + u + (y * 8 + v) * SCRWIDTH;
			accumulator[pixelAddress] = Trace( ray );
		}
	}
	// convert the floating point accumulator into pixels
	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++)
	{
		int r = min( 255, (int)(255 * accumulator[i].x) );
		int g = min( 255, (int)(255 * accumulator[i].y) );
		int b = min( 255, (int)(255 * accumulator[i].z) );
		screen->pixels[i] = (r << 16) + (g << 8) + b;
	}
}

// EOF