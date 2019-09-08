#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef wrap
#define wrap(x, lo, hi) (x < lo) ? x + (hi - lo) : (x > hi) ? x - (hi - lo) : x
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;
// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.


// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices1; // What grid cell is this particle in?
int *dev_particleGridIndices2; // What grid cell is this particle in?
int *dev_particleGridIndices3; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices2;
thrust::device_ptr<int> dev_thrust_particleGridIndices3;
thrust::device_ptr<glm::vec3> dev_thrust_pos;
thrust::device_ptr<glm::vec3> dev_thrust_vel;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}
/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		glm::vec3 rand = generateRandomVec3(time, index);
		arr[index].x = scale * rand.x;
		arr[index].y = scale * rand.y;
		arr[index].z = scale * rand.z;
	}
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
	numObjects = N;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	// LOOK-1.2 - This is basic CUDA memory management and error checking.
	// Don't forget to cudaFree in  Boids::endSimulation.
	cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

	// LOOK-1.2 - This is a typical CUDA kernel invocation.
	kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numObjects, dev_pos, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	// LOOK-2.1 computing grid params
	gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
	int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
	gridSideCount = 2 * halfSideCount;

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridInverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfSideCount;
	gridMinimum.x -= halfGridWidth;
	gridMinimum.y -= halfGridWidth;
	gridMinimum.z -= halfGridWidth;

	// TODO-2.1 TODO-2.3 - Allocate additional buffers here.
	cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

	cudaMalloc((void**)&dev_particleGridIndices1, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices1 failed!");
	cudaMalloc((void**)&dev_particleGridIndices2, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices2 failed!");
	cudaMalloc((void**)&dev_particleGridIndices3, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices3 failed!");
	cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

	cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");


	// Wrap the key/value buffers around the thrust pointers
	dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices1);
	dev_thrust_particleGridIndices2 = thrust::device_pointer_cast(dev_particleGridIndices2);
	dev_thrust_particleGridIndices3 = thrust::device_pointer_cast(dev_particleGridIndices3);

	dev_thrust_pos = thrust::device_pointer_cast(dev_pos);

	cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/


/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 new_vel(0.0f, 0.0f, 0.0f);
	int num_neighbors1 = 0, num_neighbors3 = 0;
	glm::vec3 perceived_center(0.0f, 0.0f, 0.0f), c(0.0f, 0.0f, 0.0f), perceived_velocity(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < N; i++) {
		if (i == iSelf)
			continue;
		float curr_dist = glm::distance(pos[i], pos[iSelf]);
		if (curr_dist < rule1Distance) {
			// Rule 1: boids fly towards their local perceived center of mass, which excludes 
			num_neighbors1++;
			perceived_center += pos[i];
		}
		if (curr_dist < rule2Distance) {
			// Rule 2: boids try to stay a distance d away from each other
			c -= (pos[i] - pos[iSelf]);
		}
		if (curr_dist < rule3Distance) {
			// Rule 3: boids try to match the speed of surrounding boids
			num_neighbors3++;
			perceived_velocity += vel[i];
		}
		/*rule1(curr_dist, num_neighbors1, i, iSelf, pos, vel, perceived_center);
		rule2(curr_dist, i, iSelf, pos, vel, c);
		rule3(curr_dist, num_neighbors3, i, iSelf, pos, vel, perceived_velocity);*/
	}
	// rule 1 update
	if (num_neighbors1 != 0) {
		perceived_center /= num_neighbors1;
		new_vel = (perceived_center - pos[iSelf])*rule1Scale;
	}
	// rule 2 update
	c *= rule2Scale;
	new_vel += c;
	// rule 3 update
	if (num_neighbors3 != 0) {
		perceived_velocity /= num_neighbors3;
		perceived_velocity *= rule3Scale;
		new_vel += perceived_velocity;
	}
	return new_vel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
	glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
		return;
	// Compute a new velocity based on pos and vel1
	glm::vec3 new_vel = computeVelocityChange(N, index, pos, vel1) + vel1[index];
	// Clamp the speed
	if (glm::length(new_vel) > maxSpeed)
		new_vel = (new_vel / glm::length(new_vel))*maxSpeed;
	// Record the new velocity into vel2 to not overwrite vel1 for other threads
	vel2[index] = new_vel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 thisPos = pos[index];
	thisPos += vel[index] * dt;

	// Wrap the boids around so we don't lose them
	thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
	thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
	thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

	thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
	thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
	thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

	pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(glm::vec3 pos, int gridResolution) {
	return pos.x + pos.y * gridResolution + pos.z * gridResolution * gridResolution;
}
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3 *pos, int *indices, int *gridIndices) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 correctedPos = (pos[index] + glm::vec3(scene_scale))*inverseCellWidth;
	// add indices
	gridIndices[index] = gridIndex3Dto1D(glm::floor(correctedPos), gridResolution);
	indices[index] = index;	
}
__global__ void kernComputeIndices3(int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3 *pos, int *indices, int *gridIndices1, int *gridIndices2, int *gridIndices3) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 correctedPos = (pos[index] + glm::vec3(scene_scale))*inverseCellWidth;
	// add indices
	int cell = gridIndex3Dto1D(glm::floor(correctedPos), gridResolution);
	gridIndices1[index] = cell;
	gridIndices2[index] = cell;
	gridIndices3[index] = cell;
	indices[index] = index;
}
// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids (-1), to be used with gridCellStartIndices and gridCellEndIndices
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}


__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
	int *gridCellStartIndices, int *gridCellEndIndices) {
	// Identify the start point of each cell in the gridIndices array.
  	// This is basically a parallel unrolling of a loop that goes
  	// "this index doesn't match the one before it, must be a new cell!"
  	// we need to ensure gridCellStartIndices and gridCellEndIndices have size = max number of cells in the space
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	else if (index > 0) {
		if (particleGridIndices[index - 1] != particleGridIndices[index]){ // start/end of a new block
			gridCellStartIndices[particleGridIndices[index]] = index;
			gridCellEndIndices[particleGridIndices[index-1]] = index-1;
		}
	}
	else if (index <= 0) // edge case
		gridCellStartIndices[particleGridIndices[index]] = index;
	if (index == N - 1) // edge case
		gridCellEndIndices[particleGridIndices[index]] = index;
}

__device__ void computeNeighborList(int gridResolution, float cellWidth, glm::vec3 &cell_pos, int *neighbors) {

	int gridCount = gridResolution * gridResolution*gridResolution;

	// x axis lower half
	if (std::fmod(cell_pos.x, cellWidth) < cellWidth / 2) {
		neighbors[1] = wrap(neighbors[0] - 1, 0, gridCount);
	}
	else {
		neighbors[1] = wrap(neighbors[0] + 1, 0, gridCount);
	}

	// y axis lower half
	if (std::fmod(cell_pos.y, cellWidth) < cellWidth / 2) {
		neighbors[2] = wrap(neighbors[0] - gridResolution, 0, gridCount);
		neighbors[3] = wrap(neighbors[1] - gridResolution, 0, gridCount);
	}
	else {
		neighbors[2] = wrap(neighbors[0] + gridResolution, 0, gridCount);
		neighbors[3] = wrap(neighbors[1] + gridResolution, 0, gridCount);
	}

	// z axis lower half
	if (std::fmod(cell_pos.z, cellWidth) < cellWidth / 2) {
		neighbors[4] = wrap(neighbors[0] - gridResolution * gridResolution, 0, gridCount);
		neighbors[5] = wrap(neighbors[1] - gridResolution * gridResolution, 0, gridCount);
		neighbors[6] = wrap(neighbors[2] - gridResolution * gridResolution, 0, gridCount);
		neighbors[7] = wrap(neighbors[3] - gridResolution * gridResolution, 0, gridCount);
	}
	else {
		neighbors[4] = wrap(neighbors[0] + gridResolution * gridResolution, 0, gridCount);
		neighbors[5] = wrap(neighbors[1] + gridResolution * gridResolution, 0, gridCount);
		neighbors[6] = wrap(neighbors[2] + gridResolution * gridResolution, 0, gridCount);
		neighbors[7] = wrap(neighbors[3] + gridResolution * gridResolution, 0, gridCount);
	}
}

__device__ void computeNeighborListAll(int gridResolution, float cellWidth, glm::vec3 &cell_pos, int *neighbors, int middlePos) {

	int gridCount = gridResolution * gridResolution*gridResolution;
	int n = 0;
	for (int x = -1; x <= 1; x++)
		for (int y = -1; y <= 1; y++)
			for(int z = -1;z <= 1;z++)
				neighbors[n++] = wrap(middlePos + x + y*gridResolution + z * gridResolution*gridResolution, 0, gridCount);
}

__global__ void kernUpdateVelNeighborSearchScattered(int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth, int *gridCellStartIndices, int *gridCellEndIndices,
	int *particleArrayIndices, glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) {
		return;
	}

	// useful constructs
	glm::vec3 currPos = pos[particleArrayIndices[index]];
	glm::vec3 correctedPos = (pos[particleArrayIndices[index]] + glm::vec3(scene_scale))*inverseCellWidth;
	int gridCell = gridIndex3Dto1D(glm::floor(correctedPos), gridResolution);
	glm::vec3 new_vel = vel1[particleArrayIndices[index]];
	// Identify neighboring cells
	int neighbors[8] = {gridCell};
	computeNeighborList(gridResolution, cellWidth, correctedPos, neighbors);
	// Identify all neighbors
	//int neighbors[27];
	//computeNeighborListAll(gridResolution, cellWidth, correctedPos, neighbors, gridCell);
	// Boid variables
	glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
	glm::vec3 c(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
	int num_neighbors1 = 0, num_neighbors3 = 0;

	for (int i = 0; i < 8; i++) {
		int neighborIndex = neighbors[i];
		int start = gridCellStartIndices[neighborIndex];
		int end = gridCellEndIndices[neighborIndex];
		if (start != -1 && end != -1) { // valid?
			for (int j = start; j <= end; j++) {
				if (j == index)
					continue;
				glm::vec3 otherPos = pos[particleArrayIndices[j]];
				float curr_dist = glm::length(currPos - otherPos);
				if (curr_dist < rule1Distance) {
					// Rule 1: boids fly towards their local perceived center of mass, which excludes 
					num_neighbors1++;
					perceived_center += otherPos;
				}
				if (curr_dist < rule2Distance) {
					// Rule 2: boids try to stay a distance d away from each other
					c -= (otherPos - currPos);
				}
				if (curr_dist < rule3Distance) {
					// Rule 3: boids try to match the speed of surrounding boids
					glm::vec3 otherVel = vel1[particleArrayIndices[j]];
					num_neighbors3++;
					perceived_velocity += otherVel;
				}
			}
		}
	}
	// rule 1 update
	if (num_neighbors1 != 0) {
		perceived_center /= num_neighbors1;
		new_vel += (perceived_center - currPos)*rule1Scale;
	}
	// rule 2 update
	c *= rule2Scale;
	new_vel += c;
	// rule 3 update
	if (num_neighbors3 != 0) {
		perceived_velocity /= num_neighbors3;
		perceived_velocity *= rule3Scale;
		new_vel += perceived_velocity;
	}
	// clamp & update
	if (glm::length(new_vel) > maxSpeed)
		new_vel = (new_vel / glm::length(new_vel))*maxSpeed;
	vel2[particleArrayIndices[index]] = new_vel;
}


__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	// Very similar to kernUpdateVelNeighborSearchScattered, except accessing vel & pos
	// become direct access (after we shuffle the indices)
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) {
		return;
	}

	// useful constructs
	glm::vec3 currPos = pos[index];
	glm::vec3 correctedPos = (currPos + glm::vec3(scene_scale))*inverseCellWidth;
	int gridCell = gridIndex3Dto1D(glm::floor(correctedPos), gridResolution);
	glm::vec3 new_vel = vel1[index];
	// Identify neighboring cells
	int neighbors[8] = { gridCell };
	computeNeighborList(gridResolution, cellWidth, correctedPos, neighbors);
	// Identify all neighbors
	//int neighbors[27];
	//computeNeighborListAll(gridResolution, cellWidth, correctedPos, neighbors, gridCell);
	// Boid variables
	glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
	glm::vec3 c(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
	int num_neighbors1 = 0, num_neighbors3 = 0;

	for (int i = 0; i < 8; i++) {
		int neighborIndex = neighbors[i];
		int start = gridCellStartIndices[neighborIndex];
		int end = gridCellEndIndices[neighborIndex];
		if (start != -1 && end != -1) { // valid?
			for (int j = start; j <= end; j++) {
				if (j == index)
					continue;
				glm::vec3 otherPos = pos[j];
				float curr_dist = glm::length(currPos - otherPos);
				if (curr_dist < rule1Distance) {
					// Rule 1: boids fly towards their local perceived center of mass, which excludes 
					num_neighbors1++;
					perceived_center += otherPos;
				}
				if (curr_dist < rule2Distance) {
					// Rule 2: boids try to stay a distance d away from each other
					c -= (otherPos - currPos);
				}
				if (curr_dist < rule3Distance) {
					// Rule 3: boids try to match the speed of surrounding boids
					glm::vec3 otherVel = vel1[j];
					num_neighbors3++;
					perceived_velocity += otherVel;
				}
			}
		}
	}

	// rule 1 update
	if (num_neighbors1 != 0) {
		perceived_center /= num_neighbors1;
		new_vel += (perceived_center - currPos)*rule1Scale;
	}
	// rule 2 update
	c *= rule2Scale;
	new_vel += c;
	// rule 3 update
	if (num_neighbors3 != 0) {
		perceived_velocity /= num_neighbors3;
		perceived_velocity *= rule3Scale;
		new_vel += perceived_velocity;
	}
	// clamp & update
	if (glm::length(new_vel) > maxSpeed)
		new_vel = (new_vel / glm::length(new_vel))*maxSpeed;
	vel2[index] = new_vel;
	// TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
	// except with one less level of indirection.
	// This should expect gridCellStartIndices and gridCellEndIndices to refer
	// directly to pos and vel1.
	// - Identify the grid cell that this particle is in
	// - Identify which cells may contain neighbors. This isn't always 8.
	// - For each cell, read the start/end indices in the boid pointer array.
	//   DIFFERENCE: For best results, consider what order the cells should be
	//   checked in to maximize the memory benefits of reordering the boids data.
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	// - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	//  update 
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	// ping-pong the velocity buffers
	std::swap(dev_vel1, dev_vel2);

}

void Boids::stepSimulationScatteredGrid(float dt) {
	// Compute grid indices & array indices
	dim3 fullBlocksPerGrid_gridsize((gridCellCount + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellEndIndices, -1);
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices1);

	//   Use 2x width grids.
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
		dev_thrust_particleArrayIndices);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices1,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

	// Update Velocities and Positions
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (
		numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
		dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	// - Ping-pong buffers as needed
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
	// TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	// Compute grid indices & array indices
	dim3 fullBlocksPerGrid_gridsize((gridCellCount + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernResetIntBuffer <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_gridCellStartIndices, -1);
	kernResetIntBuffer <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_gridCellEndIndices, -1);
	kernComputeIndices3 <<<fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, dev_pos, dev_particleArrayIndices, 
		dev_particleGridIndices1, dev_particleGridIndices2, dev_particleGridIndices3);

	//   Use 2x width grids.
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
		dev_thrust_particleArrayIndices);
	// sort vel and pos using the same Indices:
	thrust::sort_by_key(dev_thrust_particleGridIndices2, dev_thrust_particleGridIndices2 + numObjects,
		dev_thrust_pos);
	dev_thrust_vel = thrust::device_pointer_cast(dev_vel1);
	thrust::sort_by_key(dev_thrust_particleGridIndices3, dev_thrust_particleGridIndices3 + numObjects,
		dev_thrust_vel);
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices1,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

	// Update Velocities and Positions
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (
		numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	// - Ping-pong buffers as needed
	std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() {
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos);

	cudaFree(dev_particleArrayIndices);
	cudaFree(dev_particleGridIndices1);
	cudaFree(dev_particleGridIndices2);
	cudaFree(dev_particleGridIndices3);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_gridCellEndIndices);

}

void Boids::unitTest() {
	// test unstable sort
	int *dev_intKeys;
	int *dev_intValues;
	int N = 10;

	std::unique_ptr<int[]>intKeys{ new int[N] };
	std::unique_ptr<int[]>intValues{ new int[N] };

	intKeys[0] = 0; intValues[0] = 0;
	intKeys[1] = 1; intValues[1] = 1;
	intKeys[2] = 0; intValues[2] = 2;
	intKeys[3] = 3; intValues[3] = 3;
	intKeys[4] = 0; intValues[4] = 4;
	intKeys[5] = 2; intValues[5] = 5;
	intKeys[6] = 2; intValues[6] = 6;
	intKeys[7] = 0; intValues[7] = 7;
	intKeys[8] = 5; intValues[8] = 8;
	intKeys[9] = 6; intValues[9] = 9;

	cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

	cudaMalloc((void**)&dev_intValues, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	std::cout << "before unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// How to copy data to the GPU
	cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
	thrust::device_ptr<int> dev_thrust_values(dev_intValues);
	// Example for using thrust::sort_by_key
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// cleanup
	cudaFree(dev_intKeys);
	cudaFree(dev_intValues);
	checkCUDAErrorWithLine("cudaFree failed!");
	return;
}