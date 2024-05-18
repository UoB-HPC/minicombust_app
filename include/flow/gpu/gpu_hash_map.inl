class Hash_map
{
	protected:
		uint64_t *keys;
		uint64_t *values;
		uint64_t current_elements;
		uint64_t map_size;
	public:

	__device__ __forceinline__ void insert(uint64_t key, uint64_t value);
	__device__ __forceinline__ uint64_t find(uint64_t key);
	__device__ __forceinline__ void print();

	Hash_map(uint64_t size)
	{
		current_elements = 0;
		map_size = size;
		cudaMalloc(&keys, sizeof(uint64_t) * size);
		cudaMemset(keys, -1, sizeof(uint64_t) * size);
		cudaMalloc(&values, sizeof(uint64_t) * size);
		cudaMemset(values, -1, sizeof(uint64_t) * size);
	}
};

__device__ __forceinline__ void Hash_map::print()
{
	for(uint64_t i = 0; i < map_size; i++)
	{
		printf("%lu\n", keys[i]);
	}
}

__device__ __forceinline__ void Hash_map::insert(uint64_t key, uint64_t value)
{
	//hash the key
	uint64_t location = key % map_size;
	if(current_elements == map_size)
	{
		printf("Error: Hash table full an insert has not been completed\n");
		return;
	}
	//use linear probing to find a location
	while(values[location] != UINT64_MAX)
	{
		location = (location + 1) % map_size;
	}
	values[location] = value;
	keys[location] = key;
	current_elements += 1;
}

__device__ __forceinline__ uint64_t Hash_map::find(uint64_t key)
{
	//hash the key
	uint64_t location = key % map_size;
	while(keys[location] != key)
	{
		if(keys[location] == UINT64_MAX)
		{
			break;
			printf("Error: Failed hash table find\n");
		}
		location = (location + 1) % map_size;
	}
	return values[location];
}
