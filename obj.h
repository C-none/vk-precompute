#pragma once

#include <fstream>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <array>
#include <vector>
#include <span>
#include <stdexcept>

#include "tools.h"

class instOBJ
{

public:
	struct Vert
	{
		glm::vec3 pos;

		static constexpr auto getBindingDescription()
		{
			constexpr VkVertexInputBindingDescription bindingDescription{
				vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(Vert), VK_VERTEX_INPUT_RATE_VERTEX)};

			return bindingDescription;
		}

		static consteval auto getAttributeDescriptions()
		{
			constexpr std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{
				vertexInputAttributeDescription(0, VERTEX_BUFFER_BIND_ID, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vert, pos))};

			return attributeDescriptions;
		}
	};

	struct Instance
	{
		glm::vec3 transR1, transR2, transR3, transR4, color;

		static constexpr auto getBindingDescription()
		{
			constexpr VkVertexInputBindingDescription bindingDescription{
				vertexInputBindingDescription(INSTANCE_BUFFER_BIND_ID, sizeof(Instance), VK_VERTEX_INPUT_RATE_INSTANCE)};

			return bindingDescription;
		}

		static consteval auto getAttributeDescriptions()
		{
			constexpr std::array<VkVertexInputAttributeDescription, 5> attributeDescriptions{
				vertexInputAttributeDescription(1, INSTANCE_BUFFER_BIND_ID, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Instance, transR1)),
				vertexInputAttributeDescription(2, INSTANCE_BUFFER_BIND_ID, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Instance, transR2)),
				vertexInputAttributeDescription(3, INSTANCE_BUFFER_BIND_ID, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Instance, transR3)),
				vertexInputAttributeDescription(4, INSTANCE_BUFFER_BIND_ID, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Instance, transR4)),
				vertexInputAttributeDescription(5, INSTANCE_BUFFER_BIND_ID, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Instance, color)),
			};
			return attributeDescriptions;
		}
	};

	std::vector<Vert> vertex_data;
	std::vector<std::array<int, 3>> index_data;
	std::vector<Instance> instance_data{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {1, 1, 1}}};

	uint32_t indexCount, instanceCount, firstIndex, vertexOffset, firstInstance;
	instOBJ() = default;
	instOBJ(instOBJ &&) = default;
	instOBJ(const instOBJ &) = default;
	instOBJ &operator=(instOBJ &&) = default;
	instOBJ(const std::string &path)
	{
		using namespace std;
		ifstream in(path);

		if (!in.is_open())
		{
			throw runtime_error("failed to open file!");
		}

		for (string flag; in >> flag;)
		{
			if (flag == "v")
			{
				Vert readin{};
				for (int i = 0; i < 3; ++i)
					in >> readin.pos[i];
				vertex_data.emplace_back(readin);
			}
			else if (flag == "f")
			{
				array<int, 3> readin{};
				for (int i = 0; i < 3; ++i)
				{
					in >> readin[i];
					readin[i]--;
				}
				index_data.emplace_back(readin);
			}
			else
				in.ignore(1 << 16, '\n');
		}
		if (in.bad())
			throw runtime_error("I/O error while reading\n");
		if (vertex_data.empty())
			throw runtime_error("Empty file\n");
		in.close();
		indexCount = (uint32_t)index_data.size() * 3;
	}

	void initInstance(std::span<std::array<float, 12>> transformation, const uint32_t id)
	{
		const glm::vec3 color{((id & 0xff0000) >> 16) / 255., ((id & 0xff00) >> 8) / 255., (id & 0xff) / 255.};
		instance_data[0].color = color;
		for (auto &transform : transformation)
		{
			instance_data.emplace_back(Instance{
				*(reinterpret_cast<glm::vec3 *>(transform.data())),
				*(reinterpret_cast<glm::vec3 *>(transform.data() + 3)),
				*(reinterpret_cast<glm::vec3 *>(transform.data() + 6)),
				*(reinterpret_cast<glm::vec3 *>(transform.data() + 9)),
				color});
		}
		instanceCount = (uint32_t)instance_data.size();
	}

	void mergeData(std::vector<Vert> &vertex,
				   std::vector<std::array<int, 3>> &index,
				   std::vector<Instance> &instance)
	{
		vertexOffset = (uint32_t)vertex.size();
		firstIndex = (uint32_t)index.size() * 3;
		firstInstance = (uint32_t)instance.size();
		vertex.insert(vertex.end(), std::make_move_iterator(vertex_data.begin()), std::make_move_iterator(vertex_data.end()));
		index.insert(index.end(), std::make_move_iterator(index_data.begin()), std::make_move_iterator(index_data.end()));
		instance.insert(instance.end(), std::make_move_iterator(instance_data.begin()), std::make_move_iterator(instance_data.end()));
	}

	void draw(const VkCommandBuffer &command_buffer)
	{
		vkCmdDrawIndexed(command_buffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
	}

	// void clearmemory()
	//{
	//	vkDestroyBuffer(device, vertices.buffer, nullptr);
	//	vkFreeMemory(device, vertices.memory, nullptr);
	//	vkDestroyBuffer(device, indices.buffer, nullptr);
	//	vkFreeMemory(device, indices.memory, nullptr);
	//	vkDestroyBuffer(device, instances.buffer, nullptr);
	//	vkFreeMemory(device, instances.memory, nullptr);
	// }

	static std::vector<VkVertexInputBindingDescription> bindingDescriptions;
	static std::vector<VkVertexInputAttributeDescription> AttributeDescriptions;

	static constexpr VkPipelineVertexInputStateCreateInfo getVertexInputInfo()
	{
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		bindingDescriptions.assign({Vert::getBindingDescription(), Instance::getBindingDescription()});
		for (const auto &get_attribute_description : Vert::getAttributeDescriptions())
			AttributeDescriptions.emplace_back(get_attribute_description);
		for (const auto &get_attribute_description : Instance::getAttributeDescriptions())
			AttributeDescriptions.emplace_back(get_attribute_description);

		vertexInputInfo.vertexBindingDescriptionCount = (uint32_t)bindingDescriptions.size();
		vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)AttributeDescriptions.size();
		vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions.data();
		vertexInputInfo.pVertexAttributeDescriptions = AttributeDescriptions.data();
		return vertexInputInfo;
	}
};

