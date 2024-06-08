class Hash_map
{
	protected:
		uint64_t *keys;
		uint64_t *values;
		uint64_t current_elements;
		uint64_t map_size;
		int rank;
	public:

	__device__ __forceinline__ void insert(uint64_t key, uint64_t value);
	__device__ __forceinline__ uint64_t find(uint64_t key);
	__device__ __forceinline__ void print();

	Hash_map(int rank, uint64_t size, uint64_t *keys, uint64_t *values) : rank(rank), keys(keys), values(values)
	{
		current_elements = 0;
		map_size = size;

		cudaMemset(keys,    -1, sizeof(uint64_t) * size);
		cudaMemset(values,  -1, sizeof(uint64_t) * size);
	}
};

static __constant__ unsigned c_hash_keys[] =
{
    3499211612,  581869302, 3890346734, 3586334585,
    545404204,  4161255391, 3922919429,  949333985,
    2715962298, 1323567403,  418932835, 2350294565,
    1196140740,  809094426, 2348838239, 4264392720
};


__device__ __forceinline__ void Hash_map::print()
{
	for(uint64_t i = 0; i < map_size; i++)
	{
		printf("%lu\n", keys[i]);
	}
}

__device__ __forceinline__ uint64_t gen_hash(uint64_t key, uint64_t tries, uint64_t cur_location, uint64_t map_size)
{
	uint64_t new_location = 0;
	const uint64_t NUM_HASH_TRIES = 8;

	if (tries < NUM_HASH_TRIES)
	{
		new_location = ( (key ^ c_hash_keys[tries]) + c_hash_keys[NUM_HASH_TRIES + tries] ) & (map_size - 1);
	}
	else
	{
		new_location = (cur_location + 1) & (map_size-1);
	}

	return new_location;

}

__device__ __forceinline__ void Hash_map::insert(uint64_t key, uint64_t value)
{
	//hash the key
	const uint64_t NUM_HASH_TRIES = 8;
	uint64_t tries = 0;
	// uint64_t location = key & (map_size-1);

	uint64_t location = gen_hash(key, tries++, 0, map_size );
	uint64_t slots_checked = 0;

	if(current_elements == map_size)
	{
		printf("Rank %d Error: Hash table full an insert has not been completed\n", rank);
		return;
	}

#if __CUDA_ARCH__ >= 200 

	while( atomicCAS((unsigned long long int *)&keys[location], (unsigned long long int) UINT64_MAX, (unsigned long long int)key) != UINT64_MAX )
	// while( keys[location] != UINT64_MAX )
	{
		// location = (location + 1) & (map_size-1);
		location = gen_hash(key, tries++, location, map_size );
		slots_checked++;

		if(slots_checked == map_size + NUM_HASH_TRIES)
		{
			printf("Error: Rank %d Failed hash table insert %lu. Key location %lu. max %lu. sizeof uint %lu\n", rank, slots_checked, (unsigned long long int)keys[location], (unsigned long long int) UINT64_MAX, sizeof(unsigned long long int));
			break;
		}
	}

#else

	printf("atomicCAS is REQUIRED\n");

#endif
	// if (map_size < 50000)
	// 	printf("Rank %d Inserting (%lu, %lu) into slot %lu. Slots checked %lu. Map size %lu. sizeof %lu %lu\n", rank, key, value, location, slots_checked, map_size, sizeof(unsigned long long int), sizeof(uint64_t));
	values[location] = value;
	current_elements += 1;
}

__device__ __forceinline__ uint64_t Hash_map::find(uint64_t key)
{
	const uint64_t NUM_HASH_TRIES = 8;
	uint64_t tries = 0;
	// uint64_t location = key & (map_size-1);

	uint64_t location = gen_hash(key, tries++, 0, map_size );
	uint64_t slots_checked = 0;

	while(keys[location] != key)
	{
		// if(slots_checked == map_size + NUM_HASH_TRIES)
		// {
		// 	printf("Rank %d Error: Failed hash table find %lu. map_size %lu\n", rank, key, map_size);
		// 	break;
		// }
		location = gen_hash(key, tries++, location, map_size );
		slots_checked++;
	}
	// printf("Retrieved (%lu, %lu) into slot %lu. Slots checked %lu. Map size %lu. sizeof %lu %lu\n", keys[location], values[location], location, slots_checked, map_size, sizeof(unsigned long long int), sizeof(uint64_t));

	return values[location];
}
