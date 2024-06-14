#include "precomp.h"
#include "simdmath.h"
#include "bvh.h"

void IntersectTriAVX(RayAVX& ray, const Tri& tri, const __m256 instIdx, const __m256 triIdx)
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	__m256 edge1x = _mm256_set1_ps(edge1.x), edge1y = _mm256_set1_ps(edge1.y), edge1z = _mm256_set1_ps(edge1.z);
	__m256 edge2x = _mm256_set1_ps(edge2.x), edge2y = _mm256_set1_ps(edge2.y), edge2z = _mm256_set1_ps(edge2.z);
	//const float3 h = cross(ray.D, edge2);
	__m256 hx, hy, hz;
	crossAVX(ray.Dx8, ray.Dy8, ray.Dz8, edge2x, edge2y, edge2z, &hx, &hy, &hz);
	//const float a = dot(edge1, h);
	__m256 a = dotAVX(edge1x, edge1y, edge1z, hx, hy, hz);
	//const float f = 1 / a;
	__m256 f = _mm256_rcp_ps(a);
	//const float3 s = ray.O - tri.vertex0;
	__m256 sx = _mm256_sub_ps(ray.Ox8, _mm256_set1_ps(tri.vertex0.x));
	__m256 sy = _mm256_sub_ps(ray.Oy8, _mm256_set1_ps(tri.vertex0.y));
	__m256 sz = _mm256_sub_ps(ray.Oz8, _mm256_set1_ps(tri.vertex0.z));
	//const float u = f * dot(s, h);
	__m256 u = _mm256_mul_ps(f, dotAVX(sx, sy, sz, hx, hy, hz));
	//const float3 q = cross(s, edge1);
	__m256 qx, qy, qz;
	crossAVX(sx, sy, sz, edge1x, edge1y, edge1z, &qx, &qy, &qz);
	//const float v = f * dot(ray.D, q);
	__m256 v = _mm256_mul_ps(f, dotAVX(ray.Dx8, ray.Dy8, ray.Dz8, qx, qy, qz));
	//const float t = f * dot(edge2, q);
	__m256 t = _mm256_mul_ps(f, dotAVX(edge2x, edge2y, edge2z, qx, qy, qz));

	//if (fabs(a) < 0.00001f) return; // ray parallel to triangle
	__m256 ray_notparallel = _mm256_cmp_ps(absAVX(a), _mm256_set1_ps(0.00001f), _CMP_GE_OS);

	//if (u < 0 || u > 1) return;
	__m256 u_ge_0 = _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_GE_OS);
	__m256 u_le_1 = _mm256_cmp_ps(u, _mm256_set1_ps(1.0f), _CMP_LE_OS);
	__m256 u_01 = _mm256_and_ps(u_ge_0, u_le_1);
	//if (v < 0 || u + v > 1) return;
	__m256 v_ge_0 = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OS);
	__m256 uv_le_1 = _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_LE_OS);
	__m256 v_01 = _mm256_and_ps(v_ge_0, uv_le_1);
	//if (t > 0.0001f && t < ray.hit.t)
	__m256 t_notsmall = _mm256_cmp_ps(t, _mm256_set1_ps(0.0001f), _CMP_GT_OQ);
	__m256 t_closer = _mm256_cmp_ps(t, ray.t8, _CMP_LT_OS);
	__m256 t_validrange = _mm256_and_ps(t_notsmall, t_closer);

	__m256 fullmask = _mm256_and_ps(_mm256_and_ps(_mm256_and_ps(ray_notparallel, u_01), v_01), t_validrange);
	ray.t8 = _mm256_blendv_ps(ray.t8, t, fullmask);
	ray.u = _mm256_blendv_ps(ray.u, u, fullmask);
	ray.v = _mm256_blendv_ps(ray.v, v, fullmask);

	//printf("%d %d");

	/*
	ray.triTexture.u0 = _mm256_blendv_ps(ray.triTexture.u0, _mm256_set1_ps(tri.triex.uv0.x), fullmask);
	ray.triTexture.v0 = _mm256_blendv_ps(ray.triTexture.v0, _mm256_set1_ps(tri.triex.uv0.y), fullmask);
	ray.triTexture.u1 = _mm256_blendv_ps(ray.triTexture.u1, _mm256_set1_ps(tri.triex.uv1.x), fullmask);
	ray.triTexture.v1 = _mm256_blendv_ps(ray.triTexture.v1, _mm256_set1_ps(tri.triex.uv1.y), fullmask);
	ray.triTexture.u2 = _mm256_blendv_ps(ray.triTexture.u2, _mm256_set1_ps(tri.triex.uv2.x), fullmask);
	ray.triTexture.v2 = _mm256_blendv_ps(ray.triTexture.v2, _mm256_set1_ps(tri.triex.uv2.y), fullmask);
	*/
	ray.triIdx = _mm256_blendv_ps(ray.triIdx, triIdx, fullmask);
	ray.instIdx = _mm256_blendv_ps(ray.instIdx, instIdx, fullmask);
	//ray.hit.t = t, ray.hit.u = u,
	//ray.hit.v = v, ray.hit.instPrim = instPrim;
}

/*
void IntersectTriAVX(RayAVX& ray, const Tri& tri, const __m256 instIdx, const __m256 triIdx) {
	float Ox[8], Oy[8], Oz[8];
	float Dx[8], Dy[8], Dz[8];
	float hit_t[8];

	_mm256_storeu_ps(Ox, ray.Ox8);
	_mm256_storeu_ps(Oy, ray.Oy8);
	_mm256_storeu_ps(Oz, ray.Oz8);
	_mm256_storeu_ps(Dx, ray.Dx8);
	_mm256_storeu_ps(Dy, ray.Dy8);
	_mm256_storeu_ps(Dz, ray.Dz8);
	_mm256_storeu_ps(hit_t, ray.t8);

	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;

	for (int i = 0; i < 8; i++) {
		float3 ray_D = { Dx[i], Dy[i], Dz[i] };
		float3 ray_O = { Ox[i], Oy[i], Oz[i] };
		float3 h = cross(ray_D, edge2);
		float a = dot(edge1, h);
		if (fabs(a) < 0.00001f) continue;
		float f = 1 / a;
		float3 s = ray_O - tri.vertex0;
		float u = f * dot(s, h);
		if (u < 0 || u > 1) continue;
		float3 q = cross(s, edge1);
		float v = f * dot(ray_D, q);
		if (v < 0 || u + v > 1) continue;
		float t = f * dot(edge2, q);
		if (t > 0.0001f && t < hit_t[i]) {
			hit_t[i] = t;
		}
	}
	ray.t8 = _mm256_loadu_ps(hit_t);
}
*/

__m256 IntersectAABBAVX( const RayAVX& ray, const float3 bmin, const float3 bmax )
{
	// "slab test" ray/AABB intersection
	__m256 bminx = _mm256_set1_ps(bmin.x);
	__m256 bminy = _mm256_set1_ps(bmin.y);
	__m256 bminz = _mm256_set1_ps(bmin.z);
	__m256 bmaxx = _mm256_set1_ps(bmax.x);
	__m256 bmaxy = _mm256_set1_ps(bmax.y);
	__m256 bmaxz = _mm256_set1_ps(bmax.z);
	//float tx1 = (bmin.x - ray.O.x) * ray.rD.x;
	//float tx2 = (bmax.x - ray.O.x) * ray.rD.x;
	__m256 tx1 = _mm256_mul_ps(_mm256_sub_ps(bminx, ray.Ox8), ray.rDx8);
	__m256 tx2 = _mm256_mul_ps(_mm256_sub_ps(bmaxx, ray.Ox8), ray.rDx8);
	//float tmin = min(tx1, tx2);
	//float tmax = max(tx1, tx2);
	__m256 tmin = _mm256_min_ps(tx1, tx2);
	__m256 tmax = _mm256_max_ps(tx1, tx2);
	//float ty1 = (bmin.y - ray.O.y) * ray.rD.y;
	//float ty2 = (bmax.y - ray.O.y) * ray.rD.y;
	__m256 ty1 = _mm256_mul_ps(_mm256_sub_ps(bminy, ray.Oy8), ray.rDy8);
	__m256 ty2 = _mm256_mul_ps(_mm256_sub_ps(bmaxy, ray.Oy8), ray.rDy8);
	//tmin = max(tmin, min(ty1, ty2));
	//tmax = min(tmax, max(ty1, ty2));
	tmin = _mm256_max_ps(tmin, _mm256_min_ps(ty1, ty2));
	tmax = _mm256_min_ps(tmax, _mm256_max_ps(ty1, ty2));
	//float tz1 = (bmin.z - ray.O.z) * ray.rD.z;
	//float tz2 = (bmax.z - ray.O.z) * ray.rD.z;
	__m256 tz1 = _mm256_mul_ps(_mm256_sub_ps(bminz, ray.Oz8), ray.rDz8);
	__m256 tz2 = _mm256_mul_ps(_mm256_sub_ps(bmaxz, ray.Oz8), ray.rDz8);
	//tmin = max(tmin, min(tz1, tz2));
	//tmax = min(tmax, max(tz1, tz2));
	tmin = _mm256_max_ps(tmin, _mm256_min_ps(tz1, tz2));
	tmax = _mm256_min_ps(tmax, _mm256_max_ps(tz1, tz2));
	//if (tmax >= tmin && tmin < ray.hit.t && tmax > 0) return tmin;  else return 1e30f;
	__m256 tmax_gtet_tmin = _mm256_cmp_ps(tmax, tmin, _CMP_GE_OS);
	__m256 tmin_lt_hit = _mm256_cmp_ps(tmin, ray.t8, _CMP_LT_OS);
	__m256 tmax_gt_zero = _mm256_cmp_ps(tmax, _mm256_setzero_ps(), _CMP_GT_OS);
	return _mm256_blendv_ps(_mm256_set1_ps(1e30f), tmin, _mm256_and_ps(_mm256_and_ps(tmax_gtet_tmin, tmin_lt_hit), tmax_gt_zero));
}

void BVH::IntersectAVX( RayAVX& ray, uint instanceIdx )
{
	BVHNode* node = &bvhNode[0], * stack[64];
	uint stackPtr = 0;
	int iteration = 0;
	__m256 instanceIdx8 = _mm256_set1_ps(instanceIdx);
	while (1)
	{
		iteration++;
		if (node->isLeaf())
		{
			for (uint i = 0; i < node->triCount; i++)
			{
				IntersectTriAVX( ray, mesh->tri[triIdx[node->leftFirst + i]], instanceIdx8, _mm256_set1_ps(triIdx[node->leftFirst + i]));
			}
			if (stackPtr == 0) break;
			else node = stack[--stackPtr];
			continue;
		}
		BVHNode* child1 = &bvhNode[node->leftFirst];
		BVHNode* child2 = &bvhNode[node->leftFirst + 1];
		__m256 dist1 = IntersectAABBAVX( ray, child1->aabbMin, child1->aabbMax );
		__m256 dist2 = IntersectAABBAVX( ray, child2->aabbMin, child2->aabbMax );
		__m256 missed_child1 = _mm256_cmp_ps(dist1, _mm256_set1_ps(1e30f), _CMP_EQ_OQ);
		__m256 missed_child2 = _mm256_cmp_ps(dist2, _mm256_set1_ps(1e30f), _CMP_EQ_OQ);
		__m256 missed_both = _mm256_and_ps(missed_child1, missed_child2);
		int missed_child1_int = _mm256_movemask_ps(missed_child1);
		int missed_child2_int = _mm256_movemask_ps(missed_child2);
		int missed_both_int = _mm256_movemask_ps(missed_both);

		if (missed_both_int == 255) {
			// if all streams miss both children
			if (stackPtr == 0) break;
			else node = stack[--stackPtr];
		}
		else if (dist1.m256_f32[0] < dist2.m256_f32[0]) {
			// choose first stream as lead ray
			node = child1;
			if (missed_child2_int != 255) stack[stackPtr++] = child2;
		} 
		else {
			node = child2;
			if (missed_child1_int != 255) stack[stackPtr++] = child1;
		}
	}
}


void BVHInstance::IntersectAVX( RayAVX& ray )
{
	RayAVX backupRay = ray;
	TransformPositionAVX( ray.Ox8, ray.Oy8, ray.Oz8, invTransform );
	TransformVectorAVX( ray.Dx8, ray.Dy8, ray.Dz8, invTransform );
	ray.rDx8 = _mm256_rcp_ps(ray.Dx8);
	ray.rDy8 = _mm256_rcp_ps(ray.Dy8);
	ray.rDz8 = _mm256_rcp_ps(ray.Dz8);
	bvh->IntersectAVX( ray, idx );

	backupRay.t8 = ray.t8;
	backupRay.u = ray.u;
	backupRay.v = ray.v;
	backupRay.instIdx = ray.instIdx;
	backupRay.triIdx = ray.triIdx;
	ray = backupRay;
}

void TLAS::IntersectAVX( RayAVX& ray )
{
	ray.rDx8 = _mm256_rcp_ps(ray.Dx8);
	ray.rDy8 = _mm256_rcp_ps(ray.Dy8);
	ray.rDz8 = _mm256_rcp_ps(ray.Dz8);

	TLASNode* node = &tlasNode[0], * stack[64];
	uint stackPtr = 0;

	while (1)
	{
		if (node->isLeaf())
		{
			blas[node->BLAS].IntersectAVX( ray );
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		TLASNode* child1 = &tlasNode[node->leftRight & 0xffff];
		TLASNode* child2 = &tlasNode[node->leftRight >> 16];
		__m256 dist1 = IntersectAABBAVX( ray, child1->aabbMin, child1->aabbMax );
		__m256 dist2 = IntersectAABBAVX( ray, child2->aabbMin, child2->aabbMax );
		__m256 missed_child1 = _mm256_cmp_ps(dist1, _mm256_set1_ps(1e30f), _CMP_EQ_OQ);
		__m256 missed_child2 = _mm256_cmp_ps(dist2, _mm256_set1_ps(1e30f), _CMP_EQ_OQ);
		__m256 missed_both = _mm256_and_ps(missed_child1, missed_child2);
		int missed_child1_int = _mm256_movemask_ps(missed_child1);
		int missed_child2_int = _mm256_movemask_ps(missed_child2);
		int missed_both_int = _mm256_movemask_ps(missed_both);

		if (missed_both_int == 255) {
			// if all streams miss both children
			if (stackPtr == 0) break;
			else node = stack[--stackPtr];
		}
		else if (dist1.m256_f32[0] < dist2.m256_f32[0]) {
			// choose first stream as lead ray
			node = child1;
			if (missed_child2_int != 255) stack[stackPtr++] = child2;
		} 
		else {
			node = child2;
			if (missed_child1_int != 255) stack[stackPtr++] = child1;
		}
	}
}
