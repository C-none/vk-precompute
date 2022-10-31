#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <stdexcept>

#define DEFAULT_FENCE_TIMEOUT 100000000000
#define VERTEX_BUFFER_BIND_ID 0
#define INSTANCE_BUFFER_BIND_ID 1

#define VK_CHECK_RESULT(f)\
{VkResult res = f;\
if (res != VK_SUCCESS){\
	throw std::runtime_error(std::string("Fatal : VkResult is \"") + errorString(res) + std::string("\" in ") + std::string(__FILE__) + std::string(" at line ") + std::to_string(__LINE__) + std::string("\n"));}}


consteval VkVertexInputBindingDescription vertexInputBindingDescription(
	uint32_t binding,
	uint32_t stride,
	VkVertexInputRate inputRate)
{
	VkVertexInputBindingDescription vInputBindDescription{};
	vInputBindDescription.binding = binding;
	vInputBindDescription.stride = stride;
	vInputBindDescription.inputRate = inputRate;
	return vInputBindDescription;
}

consteval VkVertexInputAttributeDescription vertexInputAttributeDescription(
	uint32_t location,
	uint32_t binding,
	VkFormat format,
	uint32_t offset)
{
	VkVertexInputAttributeDescription vInputAttribDescription{};
	vInputAttribDescription.location = location;
	vInputAttribDescription.binding = binding;
	vInputAttribDescription.format = format;
	vInputAttribDescription.offset = offset;
	return vInputAttribDescription;
}

inline std::string errorString(VkResult errorCode)
{
	switch (errorCode)
	{
#define STR(r) case VK_ ##r: return #r
		STR(NOT_READY);
		STR(TIMEOUT);
		STR(EVENT_SET);
		STR(EVENT_RESET);
		STR(INCOMPLETE);
		STR(ERROR_OUT_OF_HOST_MEMORY);
		STR(ERROR_OUT_OF_DEVICE_MEMORY);
		STR(ERROR_INITIALIZATION_FAILED);
		STR(ERROR_DEVICE_LOST);
		STR(ERROR_MEMORY_MAP_FAILED);
		STR(ERROR_LAYER_NOT_PRESENT);
		STR(ERROR_EXTENSION_NOT_PRESENT);
		STR(ERROR_FEATURE_NOT_PRESENT);
		STR(ERROR_INCOMPATIBLE_DRIVER);
		STR(ERROR_TOO_MANY_OBJECTS);
		STR(ERROR_FORMAT_NOT_SUPPORTED);
		STR(ERROR_SURFACE_LOST_KHR);
		STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
		STR(ERROR_OUT_OF_DATE_KHR);
		STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
		STR(ERROR_VALIDATION_FAILED_EXT);
		STR(ERROR_INVALID_SHADER_NV);
		STR(SUBOPTIMAL_KHR);
		STR(ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
		STR(ERROR_NOT_PERMITTED_KHR);
		STR(ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
		STR(THREAD_IDLE_KHR);
		STR(THREAD_DONE_KHR);
		STR(OPERATION_DEFERRED_KHR);
		STR(OPERATION_NOT_DEFERRED_KHR);
		STR(ERROR_COMPRESSION_EXHAUSTED_EXT);
		STR(ERROR_OUT_OF_POOL_MEMORY_KHR);
		STR(ERROR_INVALID_EXTERNAL_HANDLE_KHR);
		STR(ERROR_FRAGMENTATION_EXT);
		STR(ERROR_INVALID_DEVICE_ADDRESS_EXT);
		STR(PIPELINE_COMPILE_REQUIRED_EXT);
		STR(ERROR_FRAGMENTED_POOL);
		STR(RESULT_MAX_ENUM);
#undef STR
	default:
	return "UNKNOWN_ERROR";
	}
}

class camera
{
public:
	glm::vec3 eye{ 0 }, forward{ 0 }, up{ 0 };
	camera(glm::vec3 pos, int64_t id) :eye(pos)
	{
		forward[(int)id / 2] = 1.f * (id % 2 ? -1 : 1);
		up[(id + 2) / 2 % 3] = 1.f * (id % 2 ? -1 : 1);
	}
	_NODISCARD std::tuple<glm::mat4, glm::mat4>getModelViewMatrix()const
	{
		glm::mat4 model;
		glm::mat4 view;
		model = glm::translate(glm::mat4(1.0f), -eye);
		view = glm::lookAt(glm::vec3(0), forward, up);
		return { model,view };
	}

};