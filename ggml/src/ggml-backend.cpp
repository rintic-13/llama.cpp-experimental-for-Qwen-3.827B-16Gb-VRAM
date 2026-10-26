// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    // the bare assert says "out of bounds" and nothing else, which is useless when the writer is a
    // scheduler path and the tensor is a wrapper whose ne/nb were rewritten. name, shape and the
    // offending range turn a crash into a line in the log.
    if (offset + size > ggml_nbytes(tensor)) {
        GGML_LOG_ERROR("%s: '%s' ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%s nbytes=%zu write=[%zu,%zu) backend=%s\n",
                __func__, tensor->name,
                (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],
                tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
                ggml_type_name(tensor->type), ggml_nbytes(tensor), offset, offset + size,
                ggml_backend_name(backend));
    }
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

// number of disjoint H2D destination cells inside one prefetch slot [GGML_SCHED_H2D_CELLS]
// the reuse distance of a physical weight destination is lcm(n_copies, n_h2d_cells) splits
#ifndef GGML_SCHED_MAX_H2D_CELLS
#define GGML_SCHED_MAX_H2D_CELLS 8
#endif

// how many splits ahead the weight copy of a split is issued [GGML_SCHED_PREFETCH_DEPTH]
// 1 reproduces the behaviour this scheduler had before: the copy of split i+1 is enqueued right after
// the compute of split i was launched. Only a hard bound lives here; the useful bound is the
// destination reuse distance minus 1, and since [GGML_SCHED_PREFETCH_DEPTH] that bound is computed per
// graph in ggml_backend_sched_reserve from the splits that really write the range. It used to be
// lcm(n_copies, n_h2d_cells) - 1, which for n_copies = 2 and one cell per slot is 1 - i.e. the old
// ceiling made a two-buffer staging (one range per graph copy) impossible to prefetch at all, and
// GGML_SCHED_H2D_CELLS was only ever bought to lift it. With the real bound the two-buffer layout is
// legal as long as the copies are far enough apart, which is what the [PREFETCH_WINDOW] line prints.
// The cap below therefore only has to be above the window a big copy needs: at ~0,42 ms per split of
// the dense 27B decode graph, a 45 MiB donor at 26-28 MiB/ms needs about 6 splits.
#ifndef GGML_SCHED_MAX_PREFETCH_DEPTH
#define GGML_SCHED_MAX_PREFETCH_DEPTH 16
#endif

// [GGML_SCHED_H2D_CHUNK] lower bound for an auto chunk size, in MiB. a chunk that is smaller than the
// weight it is carved out of is only safe while every reader of the destination waits for the last
// chunk of that weight. the split-FFN host halves (6,59 MiB at 1600 rows) are read by a consumer that
// can start before the last chunk of its own weight lands, and cutting them to 1 MiB left bytes of the
// destination that no copy had written (ffn_cpu_split_notes.md 13.55: //// garbage in the output).
// Until that hole is closed, the auto chunker does not cut anything below this size.
#ifndef GGML_SCHED_H2D_CHUNK_MIN_MIB
#define GGML_SCHED_H2D_CHUNK_MIN_MIB 8
#endif

static int ggml_sched_gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static int ggml_sched_lcm(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return a / ggml_sched_gcd(a, b) * b;
}

static size_t ggml_gcd(size_t a, size_t b) {
    while (b != 0) {
        const size_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    int copy_id; // input copy slot used by this split
    int cell_id; // H2D destination cell inside the copy slot of this split [weight_prefetch]
    // [TAG_FFN_CPU_SPLIT] see ggml_backend_cpu_split_worker below
    bool cpu_async;                 // this host split contains a MUL_MAT marked GGML_HINT_SRC0_KEEP_HOST
    int  prefix_end;                // leading nodes of this split already launched while a host split ran, 0 = none
    bool inputs_issued;             // its input copies were already issued by an earlier split's dispatch
    ggml_backend_event_t d2h_event; // fires when this split's device->host input copy has landed
    // graph view of this split
    struct ggml_cgraph graph;
};

// [TAG_FFN_CPU_SPLIT]
//
// A host split whose matmul was deliberately kept off the card has to run while the card still has work
// queued, otherwise the token pays for both sides: see the note on the device->host copy in
// copy_split_inputs. The scheduler runs splits on one thread and a host split blocks that thread, so the
// split is handed to this worker and the main thread goes on queueing the leading, independent nodes of
// the next device split (llama-graph.cpp emits the host branch of a split FFN weight before the card
// rows, which is what creates the boundary that makes those nodes independent).
//
// The worker is the only place allowed to wait. Waiting on the main thread for the copy of the activation
// would drain everything the card still had queued, the copy stream with it and the prefetch lead with
// that; waiting here costs a condition variable handover and nothing else.
//
// One job at a time on purpose: the CPU backend has a single scratch context, so two host splits running
// at once would race on it, and the whole point of the split is that its work fits into the window of one
// layer anyway.
// [CPU_SPLIT_STATS] phase timers for the CPU FFN split, to answer "where did the token time go" once the
// host half of ffn_gate/ffn_up runs next to the card half. Off unless GGML_SCHED_CPU_SPLIT_STATS=1; one
// line per 32 graphs on stderr, the accumulators reset after each line so the numbers are per window.
//
// The measured picture this is built for (27B, 1600 host rows, 22.10.2026): the serial arm pays exactly
// cpu + card per layer, the parallel arm pays max(cpu, card) + 0,130 ms of overhead per layer. Those
// 130 us per layer are what these counters localize: enqueueing the activation copy on the main thread,
// handing the split to the worker, the worker waiting for that copy, the host matmuls themselves, and the
// main thread waiting for the worker at the consumer.
static std::atomic<uint64_t> g_cpu_st_d2h_us{0};      // main: enqueue device->host activation copy + record event
static std::atomic<uint64_t> g_cpu_st_dispatch_us{0}; // main: hand the host split to the worker
static std::atomic<uint64_t> g_cpu_st_wait_us{0};     // worker: blocked until the activation copy landed
static std::atomic<uint64_t> g_cpu_st_host_us{0};     // worker: the host matmuls themselves
static std::atomic<uint64_t> g_cpu_st_join_us{0};     // main: blocked waiting for the worker
static std::atomic<uint64_t> g_cpu_st_splits{0};      // host splits seen
static std::atomic<uint64_t> g_cpu_st_graphs{0};      // graphs computed by this scheduler

// [diagnostic] per-graph budget for the [COPYPATH] probe in copy_split_inputs. Reset at every graph
// so that a run that asks the question gets an answer in every graph, not only in the first one.
// Written from the main thread only: copy_split_inputs runs on the thread that drives the scheduler.
static int g_copydiag_left = 0;

// [GGML_SCHED_H2D_CHUNK] same budget for the chunk lines: the auto chunker is a decision the measurement
// has to be able to see, and at -lv 4 the DEBUG line it normally logs is not printed. The first four
// chunks of every graph go out at INFO, which is enough to read the chunk size and the number of chunks
// out of a log without putting a line per chunk per token into a decode run.
static int g_chunkdiag_left = 0;

static bool cpu_split_stats_enabled(void) {
    static const bool on = []() {
        const char * s = getenv("GGML_SCHED_CPU_SPLIT_STATS");
        return s != NULL && atoi(s) != 0;
    }();
    return on;
}

static uint64_t cpu_split_us_now(void) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// RAII: adds the time spent in scope to one counter, costs two clock reads and one atomic add when on
struct ggml_cpu_split_timer {
    std::atomic<uint64_t> * dst;
    uint64_t t0;
    ggml_cpu_split_timer(std::atomic<uint64_t> & counter) : dst(&counter), t0(0) {
        if (cpu_split_stats_enabled()) {
            t0 = cpu_split_us_now();
        } else {
            dst = NULL;
        }
    }
    ~ggml_cpu_split_timer() {
        if (dst != NULL) {
            *dst += cpu_split_us_now() - t0;
        }
    }
};

static void cpu_split_stats_report(void) {
    if (!cpu_split_stats_enabled()) {
        return;
    }
    const uint64_t graphs = ++g_cpu_st_graphs;
    if (graphs % 32 != 0) {
        return;
    }
    const uint64_t splits  = g_cpu_st_splits.exchange(0);
    const uint64_t d2h     = g_cpu_st_d2h_us.exchange(0);
    const uint64_t dispatch= g_cpu_st_dispatch_us.exchange(0);
    const uint64_t wait    = g_cpu_st_wait_us.exchange(0);
    const uint64_t host    = g_cpu_st_host_us.exchange(0);
    const uint64_t join    = g_cpu_st_join_us.exchange(0);
    if (splits == 0) {
        return;
    }
    fprintf(stderr, "%s: [CPU_SPLIT_STATS] window=32 graphs splits=%llu d2h=%llu us dispatch=%llu us "
            "worker_wait=%llu us host=%llu us join=%llu us | per split: d2h=%.1f dispatch=%.1f "
            "wait=%.1f host=%.1f join=%.1f us\n",
            __func__, (unsigned long long) splits, (unsigned long long) d2h, (unsigned long long) dispatch,
            (unsigned long long) wait, (unsigned long long) host, (unsigned long long) join,
            (double) d2h / splits, (double) dispatch / splits, (double) wait / splits,
            (double) host / splits, (double) join / splits);
    fflush(stderr);
}

struct ggml_backend_cpu_split_worker {
    ggml_backend_t backend;

    std::mutex              mutex;
    std::condition_variable has_job;
    std::condition_variable job_finished;
    std::thread             thread;

    const struct ggml_cgraph * job_graph;
    ggml_backend_event_t       job_event;
    ggml_status                job_status;
    bool                       job_pending;
    bool                       job_running;
    bool                       stopping;

    ggml_backend_cpu_split_worker() :
        backend(NULL), job_graph(NULL), job_event(NULL), job_status(GGML_STATUS_SUCCESS),
        job_pending(false), job_running(false), stopping(false) {}

    ~ggml_backend_cpu_split_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        has_job.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
    }

    void start(ggml_backend_t cpu_backend) {
        if (thread.joinable()) {
            return;
        }
        backend = cpu_backend;

        thread = std::thread([this]() {
            for (;;) {
                const struct ggml_cgraph * graph;
                ggml_backend_event_t       event;

                {
                    std::unique_lock<std::mutex> lock(mutex);
                    has_job.wait(lock, [this]() { return job_pending || stopping; });
                    if (stopping) {
                        break;
                    }
                    graph       = job_graph;
                    event       = job_event;
                    job_pending = false;
                    job_running = true;
                }

                // this is the wait for the activation copy, and it waits only for that copy: the event was
                // recorded on the producing stream right behind the node that wrote the activation, not at
                // the end of the split that wrote it
                if (event != NULL) {
                    ggml_cpu_split_timer t_wait(g_cpu_st_wait_us);
                    ggml_backend_event_synchronize(event);
                }

                // the view is copied by value: ggml_cgraph is a small struct and its nodes pointer refers
                // to the graph the scheduler was handed, which outlives the compute call
                struct ggml_cgraph local = *graph;

                ggml_cpu_split_timer t_host(g_cpu_st_host_us);
                const ggml_status status = ggml_backend_graph_compute_async(backend, &local);

                std::lock_guard<std::mutex> lock(mutex);
                job_status  = status;
                job_running = false;
                job_finished.notify_all();
            }
        });
    }

    // dispatch a host-side split: the copy of the activation this split consumes was enqueued on the
    // producing stream when the split's inputs were copied, and the event fires when that copy has
    // landed - the worker waits for that copy and for nothing else
    void dispatch(const struct ggml_cgraph * graph, ggml_backend_event_t event) {
        ggml_cpu_split_timer t_dispatch(g_cpu_st_dispatch_us);
        ++g_cpu_st_splits;
        {
            std::lock_guard<std::mutex> lock(mutex);
            job_graph   = graph;
            job_event   = event;
            job_status  = GGML_STATUS_SUCCESS;
            job_pending = true;
        }
        has_job.notify_one();
    }

    // false when nothing was in flight, true when a job had to be waited for
    bool join(ggml_status * status) {
        ggml_cpu_split_timer t_join(g_cpu_st_join_us);
        std::unique_lock<std::mutex> lock(mutex);
        if (!job_pending && !job_running) {
            if (status != NULL) {
                *status = job_status;
            }
            return false;
        }
        job_finished.wait(lock, [this]() { return !job_pending && !job_running; });
        if (status != NULL) {
            *status = job_status;
        }
        return true;
    }
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    // per-backend split counter for weight-prefetch copy_id rotation
    // (double-buffering per backend, not a global i_split % n_copies) [weight_prefetch]
    int backend_split_count[GGML_SCHED_MAX_BACKENDS];
    // weight prefetch: each split uses its own copy slot instead of the per-graph rotating one
    bool weight_prefetch;
    // [GGML_SCHED_INPUT_SINGLE_COPY] INPUT-flagged tensors (the attention mask) carry the same bytes
    // for every consumer split, so they get ONE destination per (tensor, backend) instead of one per
    // copy_id rotation. When the consumer splits' parity is mixed (e.g. -ot shifts split ordinals),
    // the rotation allocates both destinations in the compute buffer: measured +122.25 MiB for
    // attn_inp_kq_mask at ctx 125k. The single copy is listed in the first consumer split only, so
    // it is written once per graph before any consumer runs and never overwritten mid-graph.
    bool input_single_copy;
    // disjoint H2D destination cells inside each prefetch slot. the private region of a slot is no
    // longer repacked from its beginning for every split, so a physical weight destination is reused
    // only every lcm(n_copies, n_h2d_cells) splits instead of every n_copies splits [GGML_SCHED_H2D_CELLS]
    int n_h2d_cells;
    // how many splits ahead the weight copies are kept in flight. 1 = the original one-split lookahead.
    // depth is a pure lag, never an early read: only static host weight buffers are ever prefetched
    // (the prefetchable predicate in copy_split_inputs), and each split is prefetched exactly once
    // because the window slides by one split per iteration. correctness does not depend on the depth at
    // all - a copy whose destination is still being read is parked on the CUDA backend's per-range
    // in_use event (ggml-cuda.cu). what depth changes is the state of the copy stream at a split
    // boundary: with depth 1 the stream has nothing queued, so the CPU cost of the boundary - the
    // kernel launches of the split, the event records, the slot lookup - appears on the GPU timeline as
    // an idle gap between two DMA transfers. with depth d, d transfers are already queued there [GGML_SCHED_PREFETCH_DEPTH]
    int prefetch_depth;
    // [GGML_SCHED_PREFETCH_DEPTH] the value above is clamped by lcm(n_copies, n_h2d_cells) - 1, i.e. by
    // the period of the (copy_id, cell_id) pair over ALL splits of a backend. That period is the real
    // reuse distance of the physical destination range only in a graph where every split of the backend
    // really copies weights. prefetch_depth_req is the requested depth before that clamp, and
    // prefetch_depth_eff is recomputed for every graph in ggml_backend_sched_reserve (pass D) from the
    // splits that actually write a slot range and the gaps between them. The window in
    // ggml_backend_sched_graph_compute_split uses prefetch_depth_eff; prefetch_depth stays as the
    // fallback for graphs that never reach pass D.
    int prefetch_depth_req;
    int prefetch_depth_eff;
    // [GGML_SCHED_PREFETCH_DEPTH] GGML_SCHED_PREFETCH_DEPTH unset (or 0) = auto: pass D sizes the
    // window per graph. auto is 1 - the value every depth comparison in the notes measured - unless the
    // graph was given a wider ring through GGML_SCHED_H2D_CELLS, in which case auto uses the window that
    // ring was bought for. A number in the environment is honored as before, only capped.
    bool prefetch_depth_auto;
    // [GGML_SCHED_H2D_CHUNK] GGML_SCHED_H2D_CHUNK_MIB: split the prefetch copy of a big host weight
    // into byte-range chunks and issue one chunk per split boundary of the window instead of one
    // transfer per weight. one donor of 37,85 MiB at ~25 GB/s pinned holds the single FIFO copy
    // stream for ~1,5 ms - longer than the ~1,2 ms a whole layer computes - and every small copy
    // queued behind it (the activation of the next split) waits that long: head of line. the chunks
    // ride cpy_tensor_range_async, i.e. the same copy stream and the same rotating slot as a whole
    // copy (the slot is keyed on the destination base address, ggml-cuda.cu), so the number of H2D
    // buffers, cells and events does not change. 0 (unset) = every weight copied whole, as before.
    size_t h2d_chunk_bytes;
    // [GGML_SCHED_H2D_CHUNK] GGML_SCHED_H2D_CHUNK_MIB="auto" = chunk only when the window of this graph
    // is wider than one split (chunking a weight that has to land inside one boundary is just the same
    // transfer cut into pieces that all still have to land inside that boundary), and never below
    // GGML_SCHED_H2D_CHUNK_MIN_MIB. unset or 0 = off, a number = that size. auto is opt-in because
    // cutting a donor while the window is wide has twice broken an early reader of the destination
    // (13.51/13.52 garbage, 13.58 draft acceptance 0) - the chunker is not trusted by default.
    bool h2d_chunk_auto;
    // last [PREFETCH_WINDOW] summary that was printed. reserve runs for every ubatch, so the summary is
    // printed only when the copy schedule of the graph changes, otherwise the log would get a block per
    // token.
    int dbg_pw_copies;
    int dbg_pw_reuse;
    int dbg_pw_depth;
    // how many of the leading splits are prefetched before the first compute of the graph is launched.
    // it is the pipeline fill of the window: a steady window of depth d keeps d copies ahead, and the
    // only place those d copies can be issued is at the top of the graph, so depth and burst are the
    // same number unless the fill is moved later [GGML_SCHED_PREFETCH_BURST]
    //
    // measured on the 27B dense model at 627,5 MiB of host weights per token: depth 3 with a burst of 3
    // cost 2,20 t/s against depth 1, and the loss was not in the token - the 16 weight-consuming splits
    // ran in 31,00 ms at depth 3 and 30,92 ms at depth 1, the copy engine moved the same 8158 MiB at the
    // same 24,7 GiB/s and no copy ever waited. the loss was the 2,3 ms that opened up between the end of
    // one graph's last split and the start of the next graph's first split, where 3,65 ms of burst DMA sat
    // instead of 1,05 ms. burst=1 keeps that gap at the shipped size and leaves the window itself intact.
    int prefetch_burst;
    // reusable prefetch slots: one physical buffer per (backend, copy_id) instead of a separate
    // device allocation for every copy slot of every offloaded input [GGML_SCHED_PREFETCH]
    ggml_backend_buffer_t * prefetch_slots;     // [n_backends * n_copies], or NULL
    size_t              * prefetch_slot_size;   // [n_backends * n_copies], allocated bytes
    // host weight buffers pinned (cudaHostRegister) on demand, so that the async H2D prefetch reads
    // truly pinned memory instead of being staged through pageable pages [weight_prefetch]
    void ** pinned_host_bufs;                   // [pinned_host_bufs_capacity], or NULL
    int     n_pinned_host_bufs;
    int     pinned_host_bufs_capacity;
    // when the tensor buffer is the mmap view of the whole model file, registering it in one piece
    // fails (or would pin the entire file); pin the individual tensor ranges instead, deduplicated
    // by data pointer [GGML_CUDA_PIN_RANGE]
    void ** pinned_host_ranges;                 // [pinned_host_bufs_capacity], or NULL
    int     n_pinned_host_ranges;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    // [TAG_FFN_CPU_SPLIT] runs a host split of a split FFN weight on its own thread so that the main
    // thread can go on queueing device work against it. NULL when disabled or when there is nothing to
    // overlap with. One instance per scheduler, created once: a thread per layer per token would cost
    // more than the split saves.
    struct ggml_backend_cpu_split_worker * cpu_worker;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#ifdef GGML_SCHED_DEBUG_CAUSES
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// [GGML_SCHED_LOG_VERBOSE] is the single switch for every log line this build adds on top of
// upstream llama.cpp: [ENV], [SPLIT], [ALLOC], [COMPUTE], [H2D], [SYNC], [SLOT_ASSIGN],
// [PHYS_BUF], [FALLBACK_LEAF] and [FFN_PLACEMENT]. (The macro names MOE_DIAG / MOE_LOG_* are kept
// for the lines they gate; the MoE residency subsystem that gave them their name was removed when
// the fork was trimmed to dense-only - see tmp_sched_test\ffn_cpu_split_notes.md.)
//
// 0 (unset) - off at every -lv. This build prints the same lines upstream llama.cpp prints.
// 1         - the once-per-run summaries only: [ENV], [PREFETCH_WINDOW], [PREFETCH_AUTO]. These
//             are the lines a test needs to prove the prefetch ring was in the path, and they cost
//             nothing measurable.
// 2         - everything, including the per-node [SPLIT] lines and the per-split [H2D] / [SYNC] /
//             [COMPUTE] / [ALLOC] ones.
//
// The second level exists because of where those lines sit: [SPLIT] runs per node of split_graph,
// [H2D] / [SYNC] / [COMPUTE] per split and per copy.
//
// Two costs, and the measurements separate them (tiny model, 192 decode tokens, run_log_cost.ps1,
// best of 2 interleaved runs):
//
//   -lv 4  verbose 0   436,6 t/s    273 lines
//   -lv 4  verbose 1   441,3 t/s    309 lines
//   -lv 4  verbose 2   443,6 t/s    309 lines   (the DEBUG lines are formatted but not written)
//   -lv 5  verbose 0   429,9 t/s  15658 lines
//   -lv 5  verbose 2   238,5 t/s  36344 lines   (-44,5 %)
//
// ggml_log_internal_v runs vsnprintf before the logger callback and does not consult the level
// (ggml.c:307), so a DEBUG line that -lv drops is still formatted - but at -lv 4 that costs nothing
// measurable. What costs 44 % is emitting the line: ggml_log_callback_default does fputs + fflush per
// line, so the price is paid per line actually written, not per line formatted. That is why level 2 is
// off by default and why -lv 5 is the knob to reach for last.
static int ggml_backend_sched_log_level(void) {
    static const int level = [] {
        const char * v = getenv("GGML_SCHED_LOG_VERBOSE");
        return v ? atoi(v) : 0;
    }();
    return level;
}

static bool ggml_backend_sched_log_verbose(void) {
    return ggml_backend_sched_log_level() >= 1;
}

// Wraps one added log statement. Variadic so that a whole GGML_LOG_* call is a single argument.
#define MOE_DIAG(...) \
    do { if (ggml_backend_sched_log_verbose()) { __VA_ARGS__; } } while (0)

// Shorthand for the common case: the wrapped statement is just a GGML_LOG_* call. Using these instead
// of GGML_LOG_* directly is what makes the switch above apply, so a tagged line that still uses
// GGML_LOG_* directly is a bug, not a style choice.
//
// The two spellings carry the level: an INFO summary is on at 1, a DEBUG line is on only at 2. A
// DEBUG statement that is wrapped in MOE_DIAG instead of MOE_LOG_DEBUG is a bug - it runs at level 1
// and pays for a formatted string that the logger then drops.
#define MOE_LOG_DEBUG(...) \
    do { if (ggml_backend_sched_log_level() >= 2) { GGML_LOG_DEBUG(__VA_ARGS__); } } while (0)
#define MOE_LOG_INFO(...)  MOE_DIAG(GGML_LOG_INFO(__VA_ARGS__))

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            // [TAG_FFN_CPU_SPLIT] a slice of a weight is still that weight. The split FFN matmuls use
            // ggml views of a host-resident weight, and a view has no buffer of its own: without
            // following view_src, the weight rule below never fires and the node falls through to "the
            // first backend that supports it", i.e. the card, which then copies the host half over PCIe.
            ggml_backend_buffer_t src_buf = src->buffer;
            if (src_buf == NULL && src->view_src != NULL) {
                src_buf = src->view_src->buffer;
            }
            if (src_buf != NULL && src_buf->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src_buf)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    // [TAG_FFN_CPU_SPLIT] see GGML_TENSOR_FLAG_KEEP_HOST in ggml.h
    if ((node->flags & GGML_TENSOR_FLAG_KEEP_HOST) && !ggml_backend_buft_is_host(sched->bufts[cur_backend_id])) {
        return;
    }

    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }


    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);

        // [TAG_FFN_CPU_SPLIT] see GGML_TENSOR_FLAG_KEEP_HOST in ggml.h. Left alone, the node that joins the
        // two halves of a split host matmul is handed to the card by the rule below: it carries no weight
        // for the "op follows its weight" rule to follow, the card supports the op, and the host buffer its
        // inputs live in is one the card can read, so the card is "the backend with the most supported
        // inputs". That moves the split boundary onto this node and leaves the card nothing independent to
        // run while the host split computes.
        const bool keep_host = (node->flags & GGML_TENSOR_FLAG_KEEP_HOST) != 0;

        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (keep_host && !ggml_backend_buft_is_host(sched->bufts[b])) {
                    continue;
                }
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (keep_host && !ggml_backend_buft_is_host(sched->bufts[b])) {
                    continue;
                }
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }


    // pass 5: split graph, find tensors that need to be copied
    {
        // reset per-backend split counters for weight-prefetch copy_id rotation
        for (int b = 0; b < sched->n_backends; b++) {
            sched->backend_split_count[b] = 0;
        }
        // defensive: n_h2d_cells is assigned in ggml_backend_sched_new, never let a zero reach the modulo
        if (sched->n_h2d_cells < 1) {
            sched->n_h2d_cells = 1;
        }
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        split->cpu_async = false; // [TAG_FFN_CPU_SPLIT]
        split->prefix_end = 0;    // [TAG_FFN_CPU_SPLIT]
        split->inputs_issued = false;
        if (sched->weight_prefetch) {
            const int ord = sched->backend_split_count[split->backend_id]++;
            split->copy_id = ord % sched->n_copies;
            split->cell_id = ord % sched->n_h2d_cells;
        } else {
            split->copy_id = sched->cur_copy;
            split->cell_id = 0;
        }
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                // [TAG_FFN_CPU_SPLIT] level 3 dumps every node with the backend it was given. View nodes
                // occupy indices in the split ranges even though they are skipped, and the boundary-only
                // log above cannot show where a split really ends without them. GGML_SCHED_LOG_VERBOSE=3,
                // off by default and too verbose for a timed run.
                if (ggml_backend_sched_log_level() >= 3) {
                    MOE_LOG_DEBUG("%s: [NODE] i=%d op=%s name='%s' backend=-2\n",
                        __func__, i, ggml_op_desc(node), node->name);
                }
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            if (ggml_backend_sched_log_level() >= 3) {  // [TAG_FFN_CPU_SPLIT] node dump
                MOE_LOG_DEBUG("%s: [NODE] i=%d op=%s name='%s' backend=%d\n",
                    __func__, i, ggml_op_desc(node), node->name, node_backend_id);
            }

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            // === INSTRUMENTATION: need_new_split debug ===
            if (ggml_backend_sched_log_verbose() && (need_new_split || (node_backend_id != cur_backend_id))) {
                MOE_LOG_DEBUG("%s: [SPLIT] node='%s' backend=%d cur_backend=%d need_new_split=%d\n",
                    __func__, node->name, node_backend_id, cur_backend_id, need_new_split);
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) continue;
                    if (src->buffer == NULL || src->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;
                    int src_bid = tensor_backend_id(src);
                    bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                    MOE_LOG_DEBUG("%s: [SPLIT]   src='%s' src_backend=%d supported=%d buffer=%p is_host=%d usage=%d\n",
                        __func__, src->name, src_bid, supported,
                        (void*)src->buffer,
                        ggml_backend_buffer_is_host(src->buffer),
                        (int)src->buffer->usage);
                }
            }
            // ==============================================

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                split->cpu_async = false; // [TAG_FFN_CPU_SPLIT]
                split->prefix_end = 0;    // [TAG_FFN_CPU_SPLIT]
                split->inputs_issued = false;
                if (sched->weight_prefetch) {
                    const int ord = sched->backend_split_count[node_backend_id]++;
                    split->copy_id = ord % sched->n_copies;
                    split->cell_id = ord % sched->n_h2d_cells;
                } else {
                    split->copy_id = sched->cur_copy;
                    split->cell_id = 0;
                }
                cur_backend_id = node_backend_id;

                MOE_LOG_DEBUG("%s: [SPLIT] created split[%d] backend=%d copy_id=%d cell_id=%d i_start=%d\n",
                    __func__, i_split, split->backend_id, split->copy_id, split->cell_id, split->i_start);
            }

            // [TAG_FFN_CPU_SPLIT] a matmul the CUDA backend refused to pull onto the card is the host half
            // of a split FFN weight: this split is a candidate for the worker, see
            // ggml_backend_cpu_split_worker
            if (sched->cpu_worker != NULL && !split->cpu_async &&
                    node->op == GGML_OP_MUL_MAT &&
                    ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_KEEP_HOST &&
                    ggml_backend_buft_is_host(sched->bufts[node_backend_id])) {
                split->cpu_async = true;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    } else if (sched->weight_prefetch) {
                        // each split copies into its own slot, so an input consumed by several splits
                        // has to be listed in every one of them, otherwise the slot of the later split
                        // is never written and the split reads stale data
                        bool already_listed = false;
                        for (int k = 0; k < split->n_inputs; k++) {
                            if (split->inputs[k] == src) {
                                already_listed = true;
                                break;
                            }
                        }
                        // [GGML_SCHED_INPUT_SINGLE_COPY] the single destination of an INPUT tensor is
                        // written once per graph: list it only in the first consumer split, so that a
                        // later split never overwrites data an earlier split is still reading
                        if (!already_listed && sched->input_single_copy && (src->flags & GGML_TENSOR_FLAG_INPUT)) {
                            for (int s = 0; s < i_split && !already_listed; s++) {
                                for (int k = 0; k < sched->splits[s].n_inputs; k++) {
                                    if (sched->splits[s].inputs[k] == src) {
                                        already_listed = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if (!already_listed) {
                            int n_inputs = split->n_inputs++;
                            if (n_inputs >= split->inputs_capacity) {
                                ggml_backend_sched_split_inputs_grow(split);
                            }
                            split->inputs[n_inputs] = src;
                        }
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id,
                        (sched->input_single_copy && (src->flags & GGML_TENSOR_FLAG_INPUT)) ? 0 : split->copy_id);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }


    // === INSTRUMENTATION: n_splits after split creation ===
    if (ggml_backend_sched_log_verbose()) {
        MOE_LOG_DEBUG("%s: [SPLIT] total n_splits=%d\n", __func__, sched->n_splits);
        for (int k = 0; k < sched->n_splits; k++) {
            ggml_backend_sched_split * sp = &sched->splits[k];
            const char * first_name = "(no nodes)";
            if (sp->i_start < graph->n_nodes) {
                first_name = graph->nodes[sp->i_start]->name;
            }
            const char * last_name = "(no nodes)";
            if (sp->i_end > 0 && sp->i_end <= graph->n_nodes) {
                last_name = graph->nodes[sp->i_end - 1]->name;
            }
            MOE_LOG_DEBUG("%s: [SPLIT] split[%d] backend=%d copy_id=%d i_start=%d i_end=%d n_inputs=%d first='%s' last='%s'\n",
                __func__, k, sp->backend_id, sp->copy_id,
                sp->i_start, sp->i_end, sp->n_inputs, first_name, last_name);
        }
    }
    // =====================================================

    // === INSTRUMENTATION: FFN tensor placement after all passes ===
    if (ggml_backend_sched_log_verbose()) {
        int log_count = 0;
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if (leaf->buffer == NULL || leaf->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;
            int bid = tensor_backend_id(leaf);
            if (bid <= sched->n_backends - 1) { // non-CPU
                MOE_LOG_DEBUG("%s: [FFN_PLACEMENT] leaf '%s' backend_id=%d buffer=%p is_host=%d buf_usage=%d size=%zu buft=%s backend_dev=%s\n",
                    __func__, leaf->name, bid, (void*)leaf->buffer,
                    ggml_backend_buffer_is_host(leaf->buffer),
                    (int)leaf->buffer->usage,
                    ggml_nbytes(leaf),
                    ggml_backend_buft_name(sched->bufts[bid]),
                    ggml_backend_dev_name(ggml_backend_get_device(sched->backends[bid])));
                log_count++;
            }
        }
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (node->buffer == NULL || node->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;
            int bid = tensor_backend_id(node);
            if (bid <= sched->n_backends - 1 && bid >= 0) { // non-CPU
                MOE_LOG_DEBUG("%s: [FFN_PLACEMENT] node '%s' backend_id=%d buffer=%p is_host=%d buf_usage=%d size=%zu buft=%s backend_dev=%s\n",
                    __func__, node->name, bid, (void*)node->buffer,
                    ggml_backend_buffer_is_host(node->buffer),
                    (int)node->buffer->usage,
                    ggml_nbytes(node),
                    ggml_backend_buft_name(sched->bufts[bid]),
                    ggml_backend_dev_name(ggml_backend_get_device(sched->backends[bid])));
                log_count++;
            }
        }
        if (log_count > 0) {
            MOE_LOG_DEBUG("%s: [FFN_PLACEMENT] logged %d weight tensors\n", __func__, log_count);
        }
    }
    // =============================================================

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // ---- reusable prefetch slots [GGML_SCHED_PREFETCH] -----------------------------------------
    // all weight copies of the splits that use the same (backend, copy_id) pair are aliased into one
    // persistent buffer of that slot instead of getting their own allocation. ggml-alloc skips
    // tensors that already have data set (ggml_gallocr_is_allocated), so the copies do not have to
    // be graph leafs and no longer occupy memory for the whole graph.
    if (sched->weight_prefetch) {
        const int nc = sched->n_copies;
        const int C  = sched->n_h2d_cells > 0 ? sched->n_h2d_cells : 1;
        const int n_slots = sched->n_backends * nc;

        // copies used as the weight source of GGML_OP_MUL_MAT_ID are written partially (only the
        // selected experts) and are read at the expert offsets, they are not aliased
        std::unordered_set<struct ggml_tensor *> moe_copies;
        for (int i = 0; i < sched->n_splits; i++) {
            const struct ggml_backend_sched_split * split = &sched->splits[i];
            for (int k = 0; k < split->graph.n_nodes; k++) {
                const struct ggml_tensor * node = split->graph.nodes[k];
                if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] != NULL) {
                    moe_copies.insert(node->src[0]);
                }
            }
        }

        // only static host weights that are copied in full can share a slot
        auto slot_input = [&](struct ggml_tensor * in, struct ggml_tensor * cp, ggml_backend_buffer_type_t buft) -> bool {
            ggml_backend_buffer_t buf = in->view_src ? in->view_src->buffer : in->buffer;
            return cp != NULL
                && cp->data == NULL
                && (in->flags & GGML_TENSOR_FLAG_INPUT) == 0
                && buf != NULL
                && ggml_backend_buffer_is_host(buf)
                && ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS
                // do not alias a host buft into a host slot (e.g. CPU/offload splits): pointless
                // host->host copy that only wastes RAM and blocks the host thread [weight_prefetch]
                && !ggml_backend_buft_is_host(buft)
                && moe_copies.find(cp) == moe_copies.end();
        };

        // pass A: number of splits that use each copy
        std::vector<std::unordered_map<struct ggml_tensor *, int>> n_uses(n_slots);
        for (int i = 0; i < sched->n_splits; i++) {
            const struct ggml_backend_sched_split * split = &sched->splits[i];
            const int slot = split->backend_id * nc + split->copy_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * cp = tensor_copy(split->inputs[j], split->backend_id, split->copy_id);
                if (slot_input(split->inputs[j], cp, sched->bufts[split->backend_id])) {
                    n_uses[slot][cp]++;
                }
            }
        }
        // pass B: bytes required by each slot. a copy used by more than one split has a single data
        // pointer, so it gets a fixed offset in the shared prefix of the slot. the private copies of
        // each split are packed into the cell of the slot selected by split->cell_id.
        //
        // [GGML_SCHED_H2D_CELLS] the private region is NOT repacked from the beginning for every
        // split anymore. splits of the same (backend, copy_id) pair are spread over the cells of that
        // slot, so a physical weight destination is reused only every lcm(nc, C) splits instead of every
        // nc splits. Note that copy_id and cell_id come from the SAME split counter, so the pair
        // (copy_id, cell_id) only ever takes lcm(nc, C) values: a slot only ever touches C / gcd(nc, C)
        // of its nominal cells, the others keep cell_max = 0 and are never allocated. The knob therefore
        // buys a ring of lcm(nc, C) buffers, not nc * C of them.
        // the H2D copy of split i is issued right after the compute of split i-1 was launched,
        // so with C == 1 the per-slot in_use event of the CUDA backend (ggml-cuda.cu, recorded after
        // the compute of the split that last read that address) is still unsatisfied and the copy
        // stream parks - which is what turns the prefetch into a serial H2D. with C >= nc + 1 the last
        // reader of the destination finished long ago and the copy starts immediately.
        std::vector<size_t> slot_shared(n_slots, 0);
        std::vector<size_t> cell_max((size_t) n_slots * C, 0);
        std::vector<std::unordered_set<struct ggml_tensor *>> shared_seen(n_slots);
        for (int i = 0; i < sched->n_splits; i++) {
            const struct ggml_backend_sched_split * split = &sched->splits[i];
            const int slot = split->backend_id * nc + split->copy_id;
            const int cell = (split->cell_id >= 0 && split->cell_id < C) ? split->cell_id : 0;
            const size_t align = ggml_backend_buft_get_alignment(sched->bufts[split->backend_id]);
            size_t private_sz = 0;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * cp = tensor_copy(split->inputs[j], split->backend_id, split->copy_id);
                if (!slot_input(split->inputs[j], cp, sched->bufts[split->backend_id])) {
                    continue;
                }
                const size_t sz = GGML_PAD(ggml_backend_buft_get_alloc_size(sched->bufts[split->backend_id], cp), align);
                if (n_uses[slot][cp] > 1) {
                    if (shared_seen[slot].insert(cp).second) {
                        slot_shared[slot] += sz;
                    }
                } else {
                    private_sz += sz;
                }
            }
            if (private_sz > cell_max[(size_t) slot * C + cell]) {
                cell_max[(size_t) slot * C + cell] = private_sz;
            }
        }

        // prefix offsets of the cells inside each slot: cell c of slot s starts at
        // slot_shared[s] + cell_base[s * C + c]
        std::vector<size_t> cell_base((size_t) n_slots * C, 0);
        for (int s = 0; s < n_slots; s++) {
            size_t acc = 0;
            for (int cc = 0; cc < C; cc++) {
                cell_base[(size_t) s * C + cc] = acc;
                acc += cell_max[(size_t) s * C + cc];
            }
        }
        // pass C: (re)allocate the persistent slots through the existing buft allocator
        for (int b = 0; b < sched->n_backends; b++) {
            for (int c = 0; c < nc; c++) {
                const int slot = b * nc + c;
                size_t private_sz = 0;
                for (int cc = 0; cc < C; cc++) {
                    private_sz += cell_max[(size_t) slot * C + cc];
                }
                const size_t sz = slot_shared[slot] + private_sz;
                if (sz == 0) {
                    continue;
                }
                ggml_backend_buffer_t & buf = sched->prefetch_slots[slot];
                if (buf != NULL && sched->prefetch_slot_size[slot] < sz) {
                    // a copy on the copy stream may still be writing into this slot;
                    // make sure it is done before we free the backing buffer
                    ggml_backend_synchronize(sched->backends[b]);
                    ggml_backend_buffer_free(buf);
                    buf = NULL;
                }
                if (buf == NULL) {
                    buf = ggml_backend_buft_alloc_buffer(sched->bufts[b], sz);
                    if (buf == NULL || ggml_backend_buffer_get_base(buf) == NULL) {
                        // cannot back this slot (out of memory, or a multi_buffer buft whose
                        // get_base() is NULL): gracefully disable aliasing for it so the weights
                        // fall back to normal allocation instead of crashing in pass D. [weight_prefetch]
                        sched->prefetch_slots[slot] = NULL;
                        continue;
                    }
                    // same usage as the compute buffers that hold the copies without prefetching
                    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    sched->prefetch_slot_size[slot] = sz;
                    MOE_LOG_DEBUG("%s: [PHYS_BUF] slot[%d] b=%d c=%d phys_buf=%p sz=%zu (%.2f MiB) shared=%.2f MiB cells=%d cell_sz=%.2f MiB reuse=%d buft=%s\n",
                        __func__, slot, b, c, (void*)buf, sz,
                        sz / 1024.0 / 1024.0, slot_shared[slot] / 1024.0 / 1024.0,
                        C, private_sz / 1024.0 / 1024.0, ggml_sched_lcm(nc, C),
                        ggml_backend_buft_name(sched->bufts[b]));
                    MOE_LOG_DEBUG("%s: prefetch slot[b=%d c=%d] alloc %.02f MiB (shared %.02f MiB)\n",
                        __func__, b, c, sz / 1024.0 / 1024.0, slot_shared[slot] / 1024.0 / 1024.0);
                }
            }
        }

        // pass D: point every weight copy at its place in the slot of its split. the private copies of
        // a split go into the disjoint cell selected by split->cell_id, so the copies of split i and of
        // split i + lcm(nc, C) occupy the same device memory instead of the copies of split i and i + nc
        std::vector<size_t> shared_cursor(n_slots, 0);
        std::unordered_map<struct ggml_tensor *, size_t> copy_off;

        // [GGML_SCHED_PREFETCH_DEPTH] the real H2D copy map of this graph. pass D is the only place that
        // knows both the split and the physical destination of its private copies, so it is also the
        // only place that can answer the question the depth clamp actually needs: which splits write a
        // given (slot, cell) range and how far apart they are.
        //
        // The private copies of a split are packed from the base of the cell selected by that split, so
        // all of them share one device range and one reader - the split itself. One tenant record per
        // (split, slot, cell) is therefore enough; the shared prefix of a slot is not a rotating tenant
        // at all (every split that uses such a copy writes the same bytes into the same offset).
        struct h2d_occupant {
            int    split;
            int    slot;
            int    cell;
            char * addr;
            size_t bytes;
        };
        std::vector<h2d_occupant> occupants;
        int    n_h2d_copies  = 0;
        size_t h2d_bytes     = 0;
        int    n_pref_nonslot = 0;
        // [GGML_SCHED_PREFETCH_DEPTH] how many splits of this graph run on a device backend, and how
        // many of those have a weight copy. the ratio is what the auto depth measures: see the density
        // rule below.
        int    n_dev_splits  = 0;

        // a host weight that the prefetch pass copies for a device split, whether or not it is aliased
        // into a slot. anything of this kind that is not slot_input gets its own device allocation and
        // is guarded by the scheduler's per-copy_id event instead of by a slot range, so a graph that
        // contains one keeps the static clamp.
        auto prefetch_input = [&](struct ggml_tensor * in) -> bool {
            ggml_backend_buffer_t buf = in->view_src ? in->view_src->buffer : in->buffer;
            return (in->flags & GGML_TENSOR_FLAG_INPUT) == 0
                && buf != NULL
                && ggml_backend_buffer_is_host(buf)
                && ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        };

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            const int b = split->backend_id;
            if (!ggml_backend_buft_is_host(sched->bufts[b])) {
                n_dev_splits++; // [GGML_SCHED_PREFETCH_DEPTH] a split the copy stream could be busy with
            }
            const int c = split->copy_id;
            const int slot = b * nc + c;
            const int cell = (split->cell_id >= 0 && split->cell_id < C) ? split->cell_id : 0;
            ggml_backend_buffer_t buf = sched->prefetch_slots[slot];
            if (buf == NULL) {
                // no slot for this split: its weight copies keep their own device allocations, so they
                // are not tenants of any rotating range, but they are still prefetched [GGML_SCHED_PREFETCH_DEPTH]
                if (!ggml_backend_buft_is_host(sched->bufts[b])) {
                    for (int j = 0; j < split->n_inputs; j++) {
                        if (prefetch_input(split->inputs[j])) {
                            n_pref_nonslot++;
                        }
                    }
                }
                continue;
            }
            char * base = (char *) ggml_backend_buffer_get_base(buf);
            const size_t align = ggml_backend_buft_get_alignment(sched->bufts[b]);
            size_t off_private = slot_shared[slot] + cell_base[(size_t) slot * C + cell];
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * cp = tensor_copy(split->inputs[j], b, c);
                if (!slot_input(split->inputs[j], cp, sched->bufts[b])) {
                    if (!ggml_backend_buft_is_host(sched->bufts[b]) && prefetch_input(split->inputs[j])) {
                        n_pref_nonslot++; // [GGML_SCHED_PREFETCH_DEPTH] prefetched outside the slots
                    }
                    continue;
                }
                const size_t sz = GGML_PAD(ggml_backend_buft_get_alloc_size(sched->bufts[b], cp), align);
                size_t off;
                if (n_uses[slot][cp] > 1) {
                    auto it = copy_off.find(cp);
                    if (it != copy_off.end()) {
                        off = it->second;
                    } else {
                        off = shared_cursor[slot];
                        shared_cursor[slot] += sz;
                        copy_off[cp] = off;
                    }
                } else {
                    off = off_private;
                    off_private += sz;
                }
                GGML_ASSERT(off + sz <= sched->prefetch_slot_size[slot]);
                GGML_ASSERT(ggml_backend_tensor_alloc(buf, cp, base + off) == GGML_STATUS_SUCCESS);

                // [GGML_SCHED_PREFETCH_DEPTH] one tenant record per (split, slot, cell): the private
                // copies of a split all start at the base of its cell, so they occupy one contiguous
                // device range and are read by exactly one split - the owner of the record.
                if (n_uses[slot][cp] == 1) {
                    if (occupants.empty() || occupants.back().split != i) {
                        GGML_ASSERT(occupants.empty() || occupants.back().split < i);
                        occupants.push_back({ i, slot, cell, base + off, sz });
                    } else {
                        occupants.back().bytes += sz;
                    }
                    n_h2d_copies++;
                    h2d_bytes += sz;
                }
                // === INSTRUMENTATION: slot assignment ===
                MOE_LOG_DEBUG("%s: [SLOT_ASSIGN] split=%d slot=%d b=%d c=%d cell=%d '%s' @off=%zu sz=%zu data=%p buf=%p\n",
                    __func__, i, slot, b, c, cell, cp->name, off,
                    ggml_backend_buft_get_alloc_size(sched->bufts[b], cp),
                    (void*)cp->data, (void*)buf);
                // =========================================
                if (sched->debug > 1) {
                    MOE_LOG_DEBUG("%s: prefetch slot[b=%d c=%d] split %d '%s' @ off %zu size %zu\n",
                        __func__, b, c, i, cp->name, off, ggml_backend_buft_get_alloc_size(sched->bufts[b], cp));
                }
            }
        }

        // [GGML_SCHED_PREFETCH_DEPTH] effective destination reuse distance of THIS graph.
        //
        // The static clamp in ggml_backend_sched_new is lcm(n_copies, n_h2d_cells) - 1: the period of
        // the (copy_id, cell_id) pair over ALL splits of a backend, because at split creation time the
        // scheduler does not yet know which splits will have a weight copy. It is the right number only
        // for a graph where every split of the backend really copies weights. For a graph where a few
        // splits do - the dense 27B with 6 ffn_down donors, 10 splits apart, all on the same parity of
        // copy_id - the pair repeats every 4 splits but the physical range is written and read every
        // 20, and the splits in between are not tenants of that range: they neither write it nor read
        // it, and counting them as readers makes the clamp 5x tighter than the schedule it protects.
        //
        // WAR safety does not change. The copy stream still waits on the CUDA backend's per-range
        // in_use event (ggml-cuda.cu, ggml_cuda_h2d_slot_get keyed by the destination address), which
        // is recorded after the compute of the split that last read that range. The clamp was never a
        // correctness guard - it keeps the single FIFO copy stream from queueing a transfer in front of
        // the transfers the current token is waiting for. So it has to come from the transfers that
        // exist, not from the counter that numbers the splits.
        //
        // Same rule as the static one, with the real gap: tenants of one range that are g splits apart
        // can be prefetched at most g-1 ahead, because at depth g the copy of split s is issued right
        // after the compute of split s-g is launched, which is exactly when the previous tenant's read
        // of that range is only starting.
        struct slot_cell_info {
            std::vector<int> splits;
            char *           addr    = nullptr;
            int              gap_min = INT_MAX;
        };
        std::vector<slot_cell_info> sc_info((size_t) n_slots * C);
        for (const h2d_occupant & o : occupants) {
            slot_cell_info & si = sc_info[(size_t) o.slot * C + o.cell];
            if (si.splits.empty()) {
                si.addr = o.addr;
            } else {
                si.gap_min = std::min(si.gap_min, o.split - si.splits.back());
            }
            si.splits.push_back(o.split);
        }
        int reuse_real = INT_MAX;
        for (const slot_cell_info & si : sc_info) {
            reuse_real = std::min(reuse_real, si.gap_min);
        }
        if (n_pref_nonslot > 0) {
            // prefetched weights outside the slots are guarded by the per-copy_id event, whose period is
            // the static one, so this graph keeps the static clamp [GGML_SCHED_PREFETCH_DEPTH]
            reuse_real = std::min(reuse_real, ggml_sched_lcm(nc, C));
        }

        // [GGML_SCHED_PREFETCH_DEPTH] the window for this graph.
        //
        // requested (variable set): the number from the environment, capped by the real reuse distance.
        // auto (variable unset): 1, unless the graph was given a ring that was paid for.
        //
        // Why 1 is the automatic answer: every measurement of this scheduler that compared depths came
        // out for it. dense_ab_notes 7.19/7.21 - cells=1 against cells=3 at 278,2 MiB of host weight
        // (30,70 against 30,80 t/s) and at 1239,1 MiB/token (18,30 both). dense_ab 2026-10-20 20:00 -
        // depth 3 with burst 3 cost 2,20 t/s against depth 1, and the loss was the cycle between two
        // graphs, not the token. ffn_cpu_split_notes 13.56 T2 - cells=3 with depth 5 opened the window
        // exactly as the ring model predicts (static=5 real=6 effective=5) and still cost 3,9 ms per
        // token against depth 1. depth is a lag, and a lag only pays while the copy stream is idle at
        // the split boundaries; when it is not, the queued transfers land on the compute they were
        // supposed to hide behind and take bandwidth and issue slots from it.
        //
        // The exception is a ring bought on purpose. GGML_SCHED_H2D_CELLS exists only to lift this clamp
        // (see the note in ggml_backend_sched_new): the wider ring is paid for in device staging
        // precisely so that a wider window becomes legal. When the caller bought cells, auto uses the
        // window they bought - and the auto chunker then cuts the big copies so the wide window does
        // not queue them back to back, which is the shape T2 measured without chunking.
        int          depth_eff;
        const char * depth_reason;
        const int    depth_cap = reuse_real != INT_MAX
                                 ? std::min(GGML_SCHED_MAX_PREFETCH_DEPTH, reuse_real - 1)
                                 : GGML_SCHED_MAX_PREFETCH_DEPTH;
        // a window never has to reach further ahead than the number of copies this graph contains, and
        // a range that no split reuses (reuse_real == INT_MAX: every (slot, cell) has a single tenant)
        // has no reuse distance to be capped by at all. without this second bound such a graph would be
        // given the full GGML_SCHED_MAX_PREFETCH_DEPTH [GGML_SCHED_PREFETCH_DEPTH]
        const int    auto_cap  = std::min(depth_cap, (int) occupants.size());
        int          cap_print = depth_cap;
        if (sched->prefetch_depth_auto) {
            cap_print = auto_cap;
            if (auto_cap <= 1) {
                depth_eff    = 1;
                depth_reason = reuse_real == INT_MAX ? "no reused destination" : "reuse distance 1";
            } else if (sched->n_h2d_cells > 1) {
                depth_eff    = auto_cap;
                depth_reason = "cells bought the ring";
            } else {
                depth_eff    = 1;
                depth_reason = "single-cell ring";
            }
        } else {
            depth_eff    = std::min(sched->prefetch_depth_req, GGML_SCHED_MAX_PREFETCH_DEPTH);
            if (reuse_real != INT_MAX) {
                depth_eff = std::min(depth_eff, reuse_real - 1);
            }
            depth_reason = "requested";
        }
        sched->prefetch_depth_eff = depth_eff > 1 ? depth_eff : 1;

        // [GGML_SCHED_PREFETCH_DEPTH] diagnostics: the copy schedule of this graph as it really is -
        // how many H2D weight copies it has, which device range each of them writes, the gap between
        // consecutive tenants of that range, and the depth that follows from it.
        //
        // Gated by GGML_SCHED_LOG_VERBOSE (the same switch as every other line here), not by
        // GGML_SCHED_DEBUG: -Debug in the harness sets the former. Printed once per distinct copy
        // schedule, because reserve runs for every ubatch and a per-token block would cost more than
        // the measurement (emitting a line is what is expensive, see the note on the log macros).
        if (ggml_backend_sched_log_verbose() &&
            ((int) occupants.size() != sched->dbg_pw_copies || reuse_real != sched->dbg_pw_reuse ||
             sched->prefetch_depth_eff != sched->dbg_pw_depth)) {
            sched->dbg_pw_copies = (int) occupants.size();
            sched->dbg_pw_reuse  = reuse_real;
            sched->dbg_pw_depth  = sched->prefetch_depth_eff;

            char reuse_buf[16];
            if (reuse_real == INT_MAX) {
                snprintf(reuse_buf, sizeof(reuse_buf), "none");
            } else {
                snprintf(reuse_buf, sizeof(reuse_buf), "%d", reuse_real);
            }
            MOE_LOG_INFO("%s: [PREFETCH_WINDOW] graph splits=%d | h2d weight copies=%d (%.2f MiB) in %d split(s), "
                         "prefetched outside slots=%d | reuse distance: static(lcm-1)=%d real=%s | "
                         "depth: requested=%d static=%d effective=%d | burst=%d\n",
                __func__, sched->n_splits, n_h2d_copies, h2d_bytes / 1048576.0, (int) occupants.size(),
                n_pref_nonslot, ggml_sched_lcm(nc, C) - 1, reuse_buf,
                sched->prefetch_depth_req, sched->prefetch_depth, sched->prefetch_depth_eff,
                sched->prefetch_burst);

            // [GGML_SCHED_PREFETCH_DEPTH] how the effective depth above was chosen, and what the graph
            // could have been given instead: cap is the ceiling auto used (reuse distance minus 1, capped
            // by the number of copies), so a manual depth can be set from the number instead of a guess.
            // [GGML_SCHED_H2D_CHUNK] the chunk decision is printed with it, because the two are one
            // decision: chunks only exist inside a window wider than one split.
            char chunk_mode[96];
            if (sched->h2d_chunk_auto) {
                if (sched->prefetch_depth_eff > 1) {
                    snprintf(chunk_mode, sizeof(chunk_mode), "auto %d MiB floor, window %d splits: weights above the floor get cut",
                             GGML_SCHED_H2D_CHUNK_MIN_MIB, sched->prefetch_depth_eff);
                } else {
                    snprintf(chunk_mode, sizeof(chunk_mode), "auto %d MiB floor, window 1 split: inert",
                             GGML_SCHED_H2D_CHUNK_MIN_MIB);
                }
            } else if (sched->h2d_chunk_bytes > 0) {
                snprintf(chunk_mode, sizeof(chunk_mode), "%d MiB requested", (int) (sched->h2d_chunk_bytes / 1048576u));
            } else {
                snprintf(chunk_mode, sizeof(chunk_mode), "off");
            }
            MOE_LOG_INFO("%s: [PREFETCH_AUTO] depth=%d reason=%s | %d weight copy split(s) of %d device split(s), cap=%d | chunk=%s | mode=%s\n",
                __func__, sched->prefetch_depth_eff, depth_reason, (int) occupants.size(), n_dev_splits,
                cap_print, chunk_mode, sched->prefetch_depth_auto ? "auto" : "GGML_SCHED_PREFETCH_DEPTH");

            if (ggml_backend_sched_log_level() >= 2) {
                for (int s = 0; s < n_slots; s++) {
                    for (int cc = 0; cc < C; cc++) {
                        const slot_cell_info & si = sc_info[(size_t) s * C + cc];
                        if (si.splits.empty()) {
                            continue;
                        }
                        std::string seq;
                        for (int sp : si.splits) {
                            seq += std::to_string(sp);
                            seq += ' ';
                        }
                        MOE_LOG_DEBUG("%s: [PREFETCH_WINDOW]   slot=%d (b=%d copy_id=%d) cell=%d addr=%p "
                                      "tenants=%zu splits=[%s] gap_min=%d\n",
                            __func__, s, s / nc, s % nc, cc, (void *) si.addr, si.splits.size(),
                            seq.c_str(), si.gap_min == INT_MAX ? 0 : si.gap_min);
                    }
                }
            }
        }
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id,
                (sched->input_single_copy && (input->flags & GGML_TENSOR_FLAG_INPUT)) ? 0 : split->copy_id);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    if (sched->weight_prefetch && c != split->copy_id) {
                        // with weight prefetch each split writes only its own copy slot, the copies
                        // of the other slots are unused and allocating them as leafs is what made
                        // the offloaded weights grow linearly with the number of layers
                        continue;
                    }
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    if (input_cpy->data != NULL) {
                        // aliased into a prefetch buffer, ggml-alloc must not allocate it again
                        continue;
                    }
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // === INSTRUMENTATION: fallback leafs debug ===
    if (ggml_backend_sched_log_verbose()) {
        int fallback_count = 0;
        for (int i = 0; i < graph_copy->n_leafs; i++) {
            struct ggml_tensor * l = graph_copy->leafs[i];
            if (l->data == NULL || l->buffer == NULL) {
                // This is a fallback leaf - allocated by ggml-alloc, not in a prefetch slot
                int l_bid = sched->leaf_backend_ids[i];
                MOE_LOG_DEBUG("%s: [FALLBACK_LEAF] '%s' backend=%d data=%p buffer=%p\n",
                    __func__, l->name, l_bid, (void*)l->data, (void*)l->buffer);
                fallback_count++;
            }
        }
        if (fallback_count > 0) {
            MOE_LOG_DEBUG("%s: [FALLBACK_LEAF] total %d fallback leafs (data==NULL or buffer==NULL)\n", __func__, fallback_count);
        }
    }
    // =============================================

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// === INSTRUMENTATION: alloc_splits debug ===
static void ggml_backend_sched_alloc_splits_debug(ggml_backend_sched_t sched) {
    bool ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            ids_changed = true; break;
        }
    }
    if (!ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                ids_changed = true; break;
            }
        }
    }
    MOE_LOG_DEBUG("%s: [ALLOC] backend_ids_changed=%d\n",
        __func__, ids_changed ? 1 : 0);
    MOE_LOG_DEBUG("%s: [ALLOC] n_splits=%d n_copies=%d n_h2d_cells=%d weight_prefetch=%d cur_copy=%d next_copy=%d\n",
        __func__, sched->n_splits, sched->n_copies, sched->n_h2d_cells,
        sched->weight_prefetch, sched->cur_copy, sched->next_copy);
    if (ggml_backend_sched_log_verbose()) {
        for (int k = 0; k < sched->n_splits; k++) {
            ggml_backend_sched_split * sp = &sched->splits[k];
            size_t slot_size = 0;
            if (sched->weight_prefetch) {
                const int slot = sp->backend_id * sched->n_copies + sp->copy_id;
                slot_size = sched->prefetch_slot_size[slot];
            }
            MOE_LOG_DEBUG("%s: [ALLOC] split[%d] backend=%d copy_id=%d cell_id=%d slot_size=%zu\n",
                __func__, k, sp->backend_id, sp->copy_id, sp->cell_id, slot_size);
        }
    }
}
// ===========================================

// lazily pin a host weight buffer with the split backend's device (cudaHostRegister) so that the
// async H2D prefetch reads truly pinned memory instead of being staged through pageable pages.
// the registration is done once per buffer and cached in the scheduler [weight_prefetch]
// returns true when the buffer is (or was already) registered.
static bool ggml_backend_sched_pin_host_buffer(ggml_backend_sched_t sched, ggml_backend_t split_backend, ggml_backend_buffer_t buf) {
    typedef bool (*reg_fn_t)(void *, size_t);

    ggml_backend_dev_t dev = ggml_backend_get_device(split_backend);
    if (dev == NULL) {
        return false;
    }

    void * base = ggml_backend_buffer_get_base(buf);
    if (base == NULL) {
        return false;
    }

    for (int i = 0; i < sched->n_pinned_host_bufs; i++) {
        if (sched->pinned_host_bufs[i] == base) {
            return true;  // already registered, or already known to fail
        }
    }

    if (sched->pinned_host_bufs == NULL || sched->n_pinned_host_bufs >= sched->pinned_host_bufs_capacity) {
        return false;  // tracking disabled or full: do not pin
    }

    const size_t size = ggml_backend_buffer_get_size(buf);
    // [GGML_CUDA_PIN_RANGE] with mmap the CPU tensor buffer is the view of the whole model file:
    // registering it in one piece either fails outright or pins every byte of the model into RAM.
    // skip the attempt, the caller falls back to pinning the tensor ranges that are copied H2D
    if (size > (size_t) 1024 * 1024 * 1024) {
        return false;
    }

    reg_fn_t reg = (reg_fn_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_register_host_buffer");
    if (reg == NULL) {
        return false;  // backend does not support pinning (e.g. CPU)
    }

    const bool ok = reg(base, size);

    // remember the attempt either way: a failed registration must not be retried on every token
    sched->pinned_host_bufs[sched->n_pinned_host_bufs++] = base;

    GGML_LOG_INFO("%s: [PIN] host buffer %p size=%.2f MiB on %s -> %s\n",
        __func__, base, size / 1024.0 / 1024.0, ggml_backend_dev_name(dev), ok ? "pinned" : "failed");
    return ok;
}

// [GGML_CUDA_PIN_RANGE] pin one tensor range (e.g. a -ot donor living inside the mmap view of the
// model file) so that its async H2D copy is a real pinned transfer. attempted once per data pointer
static void ggml_backend_sched_pin_host_range(ggml_backend_sched_t sched, ggml_backend_t split_backend, const ggml_tensor * t) {
    typedef bool (*reg_fn_t)(void *, size_t);

    if (sched->pinned_host_ranges == NULL || t->data == NULL) {
        return;
    }

    for (int i = 0; i < sched->n_pinned_host_ranges; i++) {
        if (sched->pinned_host_ranges[i] == t->data) {
            return;  // already attempted
        }
    }
    if (sched->n_pinned_host_ranges >= sched->pinned_host_bufs_capacity) {
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(split_backend);
    if (dev == NULL) {
        return;
    }
    reg_fn_t reg = (reg_fn_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_register_host_buffer");
    if (reg == NULL) {
        return;
    }

    const size_t size = ggml_nbytes(t);
    const bool ok = reg(t->data, size);

    sched->pinned_host_ranges[sched->n_pinned_host_ranges++] = t->data;

    GGML_LOG_INFO("%s: [PIN] tensor '%s' range %p size=%.2f MiB on %s -> %s\n",
        __func__, t->name, t->data, size / 1024.0 / 1024.0, ggml_backend_dev_name(dev), ok ? "pinned" : "failed");
}

// [TAG_FFN_CPU_SPLIT] did this split produce the tensor that a later split reads - either in place, or
// through the copy that pass 5 put in its place?
//
// Two things hide the dependency from a plain pointer comparison, both measured on
// tmp_sched_test\tiny-llama-q4_0:
//   - the card cannot read a host split's result where it stands. On Windows the CUDA buffer type does not
//     support host buffers (there is no direct host access), so pass 5 replaced the read with a copy of it:
//     the consumer's node points at the copy tensor and the original is only reachable through the split's
//     input list, which tensor_copy maps to that copy;
//   - a view of a host-resident weight is a node of the host split too, and the card reads a copy of it as
//     well, but that is not a dependency on the host compute - the bytes were already there before the split
//     ran and their copy can be issued while the worker is still going. View nodes are therefore not treated
//     as producers.
static bool ggml_backend_sched_split_produced(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split,
                                              struct ggml_tensor * src, const struct ggml_backend_sched_split * consumer) {
    for (int m = 0; m < split->graph.n_nodes; m++) {
        const struct ggml_tensor * p = split->graph.nodes[m];
        if (ggml_is_view_op(p->op)) {
            continue;
        }
        if (p == src) {
            return true;
        }
    }

    for (int j = 0; j < consumer->n_inputs; j++) {
        struct ggml_tensor * in = consumer->inputs[j];
        if (in->view_src != NULL || tensor_copy(in, consumer->backend_id, consumer->copy_id) != src) {
            continue;
        }
        for (int m = 0; m < split->graph.n_nodes; m++) {
            const struct ggml_tensor * p = split->graph.nodes[m];
            if (!ggml_is_view_op(p->op) && p == in) {
                return true;
            }
        }
    }

    return false;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    g_copydiag_left = 4;  // [diagnostic] see the [COPYPATH] probe in copy_split_inputs below
    g_chunkdiag_left = 4; // [GGML_SCHED_H2D_CHUNK] see the [H2D-CHUNK] lines below

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;


    int prev_backend_id = -1;
    int prev_copy_id = -1;


    // copy the inputs of a split into the copy slot assigned to that split
    // [GGML_SCHED_H2D_CHUNK] per-split chunk plan: split_chunk_n[k] is how many chunks the big
    // prefetchable weights of split k are cut into (0/1 = copied whole), split_chunk_done[k] how
    // many of them the window has already issued. both are filled by the chunk 0 call - the first
    // touch of the split by the prefetch window - and consumed by the continuation pass and by the
    // flush in front of the split's launch.
    std::vector<int> split_chunk_n(sched->n_splits, 0);
    std::vector<int> split_chunk_done(sched->n_splits, 0);

    auto copy_split_inputs = [&](int split_id, bool weights_only, int chunk_j = -1) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        const int copy_id = split->copy_id;

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id,
                (sched->input_single_copy && (input->flags & GGML_TENSOR_FLAG_INPUT)) ? 0 : copy_id);

            // only static host weights can be copied before the split runs, the other inputs are
            // produced by earlier splits and copying them early would read stale data.
            ggml_backend_buffer_t pbuf = input->view_src ? input->view_src->buffer : input->buffer;
            bool prefetchable = !(input->flags & GGML_TENSOR_FLAG_INPUT) &&
                                pbuf != NULL &&
                                ggml_backend_buffer_is_host(pbuf) &&
                                ggml_backend_buffer_get_usage(pbuf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;

            if (prefetchable) {
                for (int i = 0; i < split->graph.n_nodes; i++) {
                    const struct ggml_tensor * node = split->graph.nodes[i];
                    if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy) {
                        prefetchable = false;
                        break;
                    }
                }
            }

            if (sched->weight_prefetch && prefetchable != weights_only) {
                continue;
            }

            // [diagnostic] this input survived the pass filter and is about to be copied. For the -ot
            // donor weights the reading is: a line with weights_only=1 means the prefetch pass is
            // copying it now (the [PIN] line belongs to the same decision); a line with weights_only=0
            // means the normal pass is copying it, i.e. the prefetch pass did not cover this split; and
            // no line at all in a graph means the tensor is not an input of any CUDA split - its
            // MUL_MAT was assigned to a CPU split and it is read straight from the mmap buffer.
            if (g_copydiag_left > 0 && input->name && strstr(input->name, "ffn_down.weight") != NULL) {
                g_copydiag_left--;
                MOE_LOG_INFO("%s: [COPYPATH] '%s' split=%d weights_only=%d prefetchable=%d split_backend=%s cpy_async=%d\n",
                        __func__, input->name, split_id, weights_only, prefetchable,
                        ggml_backend_name(split_backend),
                        split_backend->iface.cpy_tensor_async != NULL);
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->input_single_copy) {
                    // [GGML_SCHED_INPUT_SINGLE_COPY] the one destination may still be read by the
                    // previous graph: wait for the last split that actually reads it (scan the plan
                    // backwards), not for every copy slot - waiting on both slots drains the whole
                    // compute pipeline at every graph boundary (measured: -10 t/s)
                    int last_cid = copy_id;
                    for (int s = sched->n_splits - 1; s >= 0; s--) {
                        const struct ggml_backend_sched_split * sp = &splits[s];
                        if (sp->backend_id != split_backend_id) {
                            continue;
                        }
                        bool reads = false;
                        for (int i = 0; i < sp->graph.n_nodes && !reads; i++) {
                            const struct ggml_tensor * n = sp->graph.nodes[i];
                            for (int j = 0; j < GGML_MAX_SRC; j++) {
                                if (n->src[j] == input_cpy) {
                                    reads = true;
                                    break;
                                }
                            }
                        }
                        if (reads) {
                            last_cid = sp->copy_id;
                            break;
                        }
                    }
                    if (sched->events[split_backend_id][last_cid] != NULL) {
                        ggml_backend_event_synchronize(sched->events[split_backend_id][last_cid]);
                    } else {
                        ggml_backend_synchronize(split_backend);
                    }
                } else if (sched->events[split_backend_id][copy_id] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][copy_id]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][copy_id] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][copy_id]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }


                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    if (ggml_nelements(ids_tensor) == 0) {
                        continue;
                    }

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, copy_id)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        // wait for the backend that produced the ids, then read them (the upstream way;
                        // the event-based read lived in the removed MoE residency path)
                        ggml_backend_synchronize(ids_backend);
                        ggml_backend_tensor_get(ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else if (split->cpu_async && sched->cpu_worker != NULL &&
                        input_backend->iface.get_tensor_async != NULL &&
                        input_cpy->data != NULL &&
                        (pbuf == NULL || !ggml_backend_buffer_is_host(pbuf))) {
                    // [TAG_FFN_CPU_SPLIT] direction device -> host: the activation that the host half of a
                    // split FFN weight has to matmul.
                    //
                    // The path below cannot be used here: ggml_backend_tensor_copy_async has no async path
                    // for a host destination, so it synchronizes the producing backend and blocks - in the
                    // middle of a token that drains everything the card still had queued, the copy stream
                    // with it, and the prefetch lead with that.
                    //
                    // Instead the copy is enqueued on the producing stream, which puts it right behind the
                    // node that wrote the activation and in front of everything the card is about to be
                    // given, into the pinned host buffer the scheduler already reserves for the host
                    // backend (sched_reserve: CUDA_Host compute buffer). The one that waits for it is the
                    // worker thread that is going to read it, so the main thread never stops queueing.
                    if (split->d2h_event == NULL) {
                        split->d2h_event = ggml_backend_event_new(ggml_backend_get_device(input_backend));
                    }

                    {
                        ggml_cpu_split_timer t_d2h(g_cpu_st_d2h_us);
                        ggml_backend_tensor_get_async(input_backend, input, input_cpy->data, 0, ggml_nbytes(input));

                        if (split->d2h_event != NULL) {
                            // recorded after each copy of this split, so waiting on it waits for all of them
                            ggml_backend_event_record(split->d2h_event, input_backend);
                        } else {
                            // no event on that device: fall back to the blocking behaviour
                            ggml_backend_synchronize(input_backend);
                        }
                    }

                    MOE_LOG_DEBUG("%s: [D2H] split=%d copy_id=%d '%s' -> '%s' bytes=%zu event=%p\n",
                        __func__, split_id, copy_id, input->name, input_cpy->name,
                        ggml_nbytes(input), (void*)split->d2h_event);
                } else {
                    // [GGML_SCHED_H2D_CHUNK] the prefetch pass copies a big host weight in byte-range
                    // chunks, one chunk per split boundary of the window (chunk_j = -1 is chunk 0, the
                    // window's first touch of the split; chunk_j >= 1 is a continuation issued at a
                    // later boundary). the chunks go through cpy_tensor_range_async - the same copy
                    // stream and the same rotating slot as a whole copy: the slot is keyed on the base
                    // address of the destination, the compute side waits on the slot's ready event at
                    // the split's launch (after the last chunk was enqueued, the event captures the
                    // last record), and the next tenant of the slot waits on in_use before its first
                    // chunk. no new buffers, cells or events. what changes is only how long one
                    // transfer can block everything queued behind it in the FIFO: one chunk, not one
                    // weight. the chunk size is never below nbytes/depth_eff, so the weight always
                    // fits the window that is allowed to reach this split.
                    const size_t nbytes_w = ggml_nbytes(input);
                    const int    depth    = std::min(std::max(sched->prefetch_depth_eff, 1), GGML_SCHED_MAX_PREFETCH_DEPTH);
                    // [GGML_SCHED_H2D_CHUNK] the size to ask for. manual = the knob. auto = the floor,
                    // and only when the window of this graph is wider than one split: at depth 1 every
                    // chunk of a weight still has to land inside the single boundary before that split
                    // runs, so cutting the weight changes nothing but the number of transfers. the floor
                    // is what keeps a weight that is smaller than it whole - which is also what keeps the
                    // split-FFN host halves out of the hole described at GGML_SCHED_H2D_CHUNK_MIN_MIB.
                    size_t chunk_req = sched->h2d_chunk_bytes;
                    if (sched->h2d_chunk_auto) {
                        chunk_req = depth > 1 ? (size_t) GGML_SCHED_H2D_CHUNK_MIN_MIB * 1048576u : 0;
                    }
                    bool chunkable = chunk_req > 0 && weights_only && prefetchable &&
                                     split_backend->iface.cpy_tensor_range_async != NULL &&
                                     input->data != NULL && input_cpy->data != NULL &&
                                     nbytes_w > chunk_req;
                    if (!chunkable && chunk_j >= 1) {
                        continue;  // continuation pass: this weight went whole at chunk 0 already
                    }
                    if (chunkable) {
                        const size_t cs    = std::max(chunk_req, (nbytes_w + depth - 1) / depth);
                        const int    n_in  = (int) ((nbytes_w + cs - 1) / cs);
                        if (split_chunk_n[split_id] < n_in) {
                            split_chunk_n[split_id] = n_in;
                        }
                        const int j = chunk_j >= 0 ? chunk_j : 0;
                        bool      done_for_this = false;
                        if (j >= n_in || (j == 0 && split_chunk_done[split_id] > 0)) {
                            done_for_this = true;  // this input is complete already, nothing to copy
                        } else {
                            if (j == 0 && split_backend->iface.cpy_tensor_async != NULL) {
                                if (!ggml_backend_sched_pin_host_buffer(sched, split_backend, pbuf)) {
                                    ggml_backend_sched_pin_host_range(sched, split_backend, input);
                                }
                            }
                            const size_t off = (size_t) j * cs;
                            const size_t len = std::min(cs, nbytes_w - off);
                            if (split_backend->iface.cpy_tensor_range_async(
                                    input_backend, split_backend, input, input_cpy, off, off, len)) {
                                done_for_this = true;
                                if (split_chunk_done[split_id] < j + 1) {
                                    split_chunk_done[split_id] = j + 1;
                                }
                                if (ggml_backend_sched_log_verbose() && g_chunkdiag_left > 0) {
                                    g_chunkdiag_left--;
                                    MOE_LOG_INFO("%s: [H2D-CHUNK] split=%d copy_id=%d '%s' chunk=%d/%d off=%zu bytes=%zu (chunk size %zu MiB, depth %d)\n",
                                        __func__, split_id, copy_id, input->name, j, n_in, off, len, cs / 1048576u, depth);
                                } else {
                                    MOE_LOG_DEBUG("%s: [H2D-CHUNK] split=%d copy_id=%d '%s' chunk=%d/%d off=%zu bytes=%zu\n",
                                        __func__, split_id, copy_id, input->name, j, n_in, off, len);
                                }
                            }
                        }
                        if (done_for_this) {
                            continue;
                        }
                        chunkable = false;  // the range path refused: fall back to the whole copy
                    }
                    // pin the host weight buffer once so that the async copy below is a real
                    // pinned H2D transfer instead of pageable staging [weight_prefetch]
                    if (prefetchable && split_backend->iface.cpy_tensor_async != NULL) {
                        if (!ggml_backend_sched_pin_host_buffer(sched, split_backend, pbuf)) {
                            ggml_backend_sched_pin_host_range(sched, split_backend, input);
                        }
                    }
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    // === INSTRUMENTATION: H2D timeline ===
                    MOE_LOG_DEBUG("%s: [H2D] split=%d copy_id=%d '%s' -> '%s' async=%d event=%p\n",
                        __func__, split_id, copy_id, input->name, input_cpy->name,
                        (split_backend->iface.cpy_tensor_async != NULL),
                        (void*)sched->events[split_backend_id][copy_id]);
                    // =====================================
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][copy_id] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][copy_id]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

    };

    // prefetch the leading splits (GGML_SCHED_PREFETCH_BURST of them), then keep the weight copies of
    // the splits that are still prefetch_depth splits ahead in flight while the current split computes
    // on another slot
    //
    // The original note here claimed the burst at the top of the graph was what gave the copy stream work
    // to run on. Measured on the 27B dense model that is only half true, and the other half is a cost:
    // a burst of 3 tightened the copy stream inside the token (idle between transfers 2,41 -> 1,51 ms per
    // token) and simultaneously put 3,65 ms of DMA - instead of 1,05 ms - into the cycle between two
    // graphs, where the logits are read, the token is sampled and the next graph is built. That cycle grew
    // from 0,44 to 2,76 ms, the GPU had nothing to run in it, and the token cost 2,49 ms more [dense_ab
    // 2026-10-20 20:00, arm I against arm C]. The fill is therefore configurable: burst=1 pays for the
    // window with a single leading transfer, the rest of the depth is filled at the first split boundary,
    // where the compute of split 0 is already in flight and can hide it.
    int prefetched_upto = -1;
    // [GGML_SCHED_PREFETCH_DEPTH] how far ahead the window reaches for THIS graph. sized by the reuse
    // distance of the ranges that the graph really writes (ggml_backend_sched_reserve, pass D), not by
    // the period of the split counter.
    int prefetch_window = 1;
    if (sched->weight_prefetch && sched->n_splits > 0) {
        // defensive: both are assigned in ggml_backend_sched_new, never let a zero reach the window
        if (sched->prefetch_depth < 1) {
            sched->prefetch_depth = 1;
        }
        int depth = sched->prefetch_depth_eff;
        if (depth < 1 || (!sched->prefetch_depth_auto && depth > sched->prefetch_depth_req)) {
            // never wider than what was asked for, whatever reserve computed [GGML_SCHED_PREFETCH_DEPTH]
            depth = sched->prefetch_depth;
        }
        prefetch_window = depth;
        const int burst = std::min(std::max(sched->prefetch_burst, 1), depth);
        prefetched_upto = std::min(burst, sched->n_splits) - 1;
        for (int i = 0; i <= prefetched_upto; i++) {
            copy_split_inputs(i, true);
        }
    }

    // [TAG_FFN_CPU_SPLIT] the split that is running on the worker, and the first split that cannot have
    // its inputs copied before the worker is done with it. Both are -1 whenever nothing is in flight.
    int worker_split   = -1;
    int worker_join_at = -1;

    auto worker_join = [&](void) {
        if (sched->cpu_worker != NULL && worker_split >= 0) {
            sched->cpu_worker->join(NULL);
            worker_split   = -1;
            worker_join_at = -1;
        }
    };

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // [TAG_FFN_CPU_SPLIT] the result of a host split reaches the card through a copy of it, and that
        // copy cannot be issued before the host threads have written the result
        if (worker_split >= 0 && split_id == worker_join_at) {
            ggml_status wstatus = GGML_STATUS_SUCCESS;
            sched->cpu_worker->join(&wstatus);

            MOE_LOG_DEBUG("%s: [CPU_SPLIT] joined the worker of split=%d before split=%d\n",
                __func__, worker_split, split_id);

            worker_split   = -1;
            worker_join_at = -1;

            if (wstatus != GGML_STATUS_SUCCESS) {
                return wstatus;
            }
        }

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            // wait on the previous split's actual event on the previous backend, not a slot key it never used
            int prev_c = prev_copy_id >= 0 ? prev_copy_id : split->copy_id;
            if (sched->events[prev_backend_id][prev_c] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][prev_c]);
                // === INSTRUMENTATION: sync wait ===
                MOE_LOG_DEBUG("%s: [SYNC] waited on event backend=%d copy_id=%d for split=%d\n",
                    __func__, prev_backend_id, prev_c, split_id);
                // ===================================
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // [GGML_SCHED_H2D_CHUNK] the split is about to run: every chunk of its weights has to be in
        // the slot. the pacing above schedules chunk n-1 at the boundary right before this launch, so
        // the loop normally issues nothing - it is the guarantee, not the mechanism. (early splits of
        // the graph, whose n boundaries do not exist before the launch, are exactly what it covers.)
        if ((sched->h2d_chunk_bytes > 0 || sched->h2d_chunk_auto) && split_chunk_n[split_id] > split_chunk_done[split_id]) {
            for (int j = split_chunk_done[split_id]; j < split_chunk_n[split_id]; j++) {
                copy_split_inputs(split_id, true, j);
            }
        }

        // [TAG_FFN_CPU_SPLIT] a dispatching split may already have issued this split's input copies, which
        // is what let it launch the leading part of this split while the host threads ran. copying them a
        // second time here would overwrite bytes that that early compute is still reading
        if (!split->inputs_issued) {
            if (!sched->weight_prefetch) {
                copy_split_inputs(split_id, false);
            } else {
                // the inputs which could not be prefetched are copied at the original point
                copy_split_inputs(split_id, false);
            }
        }


        // [TAG_FFN_CPU_SPLIT] hand a host split of a split FFN weight to the worker, then give the card
        // something to run against it.
        //
        // Why this is possible at all: llama-graph.cpp makes the host half of the split weight the leading
        // rows and src[0] of the concatenation, so the split boundary falls on the norm that produced the
        // activation and the split behind it starts with the card's own gate/up matmuls - nodes that read
        // nothing the host split produces. Those leading nodes are launched here while the host threads run;
        // the rest of the same split, from the first node that reads the copy of the host result, is launched
        // in the next iteration, after the join above. Without this the host threads would run with the card
        // idle and the token would pay for both sides of the split.
        bool on_worker = false;

        if (sched->cpu_worker != NULL && split->cpu_async && !sched->callback_eval &&
                split->prefix_end == 0 && worker_split < 0 && split->graph.n_nodes > 0) {
            const int next_id = split_id + 1;

            int prefix           = -1;
            int join_at          = -1;
            bool copy_dependency = false;

            // the overlap only exists when the split that consumes the host result runs on a device: with
            // a host split behind there is nothing for the card to run against, and handing the work to
            // the worker would add a thread handover on top of the same serialized time. requiring the
            // consumer to be the immediately next split is also what keeps the early launch safe: a device
            // split further ahead could read something an intermediate host split produces.
            if (next_id < sched->n_splits &&
                    !ggml_backend_buft_is_host(sched->bufts[splits[next_id].backend_id])) {
                join_at = next_id;

                struct ggml_backend_sched_split * next = &splits[join_at];

                // the first node of the consumer that reads a computed result of this split. everything
                // before it is independent of the host threads and can be queued on the card while they run.
                // the nodes of a split are in dependency order, so the first hit is the earliest one.
                for (int k = 0; k < next->graph.n_nodes; k++) {
                    const struct ggml_tensor * node = next->graph.nodes[k];

                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        struct ggml_tensor * src = node->src[s];
                        if (src == NULL) {
                            continue;
                        }
                        if (src->view_src != NULL) {
                            src = src->view_src;
                        }
                        if (ggml_backend_sched_split_produced(sched, split, src, next)) {
                            prefix = k;
                            break;
                        }
                    }

                    if (prefix >= 0) {
                        break;
                    }
                }
            }

            // does the consumer read the host result through a copy of it? on this machine the answer is
            // always yes when a host split feeds the card, and it decides which of the consumer's input
            // copies may be issued before the worker is done - see the early issue below
            if (join_at >= 0) {
                struct ggml_backend_sched_split * next = &splits[join_at];

                for (int j = 0; j < next->n_inputs && !copy_dependency; j++) {
                    if (next->inputs[j]->view_src != NULL) {
                        continue;
                    }
                    for (int m = 0; m < split->graph.n_nodes; m++) {
                        const struct ggml_tensor * p = split->graph.nodes[m];
                        if (!ggml_is_view_op(p->op) && p == next->inputs[j]) {
                            copy_dependency = true;
                            break;
                        }
                    }
                }

            }

            if (join_at < 0) {
                MOE_LOG_DEBUG("%s: [CPU_SPLIT] split=%d kept on the main thread: no device split behind it\n",
                    __func__, split_id);
            } else {
                struct ggml_backend_sched_split * next = &splits[join_at];

                // the head reads the inputs of the consumer, so those copies have to be in flight before it
                // is queued. with weight_prefetch the weights are the sliding window's business and are
                // already on their way; without the window they are issued here, and only the weights when
                // the consumer also reads something the worker is about to produce
                if (prefix < 0) {
                    prefix = 0;
                }

                if (sched->weight_prefetch) {
                    if (join_at > prefetched_upto) {
                        copy_split_inputs(join_at, true);
                        prefetched_upto = join_at;
                    }
                } else if (prefix > 0 && !copy_dependency) {
                    // the window is off, so this is the only place the consumer's copies can be issued early.
                    copy_split_inputs(join_at, false);
                    next->inputs_issued = true;
                } else if (prefix > 0 && copy_dependency) {
                    // the copy of the host result must not go before the worker has written it, but the
                    // consumer's static weights can, and those are the bulk of its input traffic. inputs_issued
                    // stays false, so the rest is copied in the consumer's own iteration, after the join
                    copy_split_inputs(join_at, true);
                } else {
                    // nothing independent to launch, so there is nothing to issue early either
                    prefix = 0;
                }


                // dispatched before the head is queued: the worker starts as soon as the copy of the
                // activation lands, and the head lands on the card behind that same copy
                sched->cpu_worker->start(split_backend);
                sched->cpu_worker->dispatch(&split->graph, split->d2h_event);

                worker_split   = split_id;
                worker_join_at = join_at;
                on_worker      = true;

                if (prefix > 0) {
                    struct ggml_cgraph head = ggml_graph_view(&next->graph, 0, prefix);

                    MOE_LOG_DEBUG("%s: [CPU_SPLIT] split=%d handed to the worker, %d of %d leading nodes of"
                                  " split=%d launched early\n",
                        __func__, split_id, prefix, next->graph.n_nodes, join_at);

                    enum ggml_status ec = ggml_backend_graph_compute_async(sched->backends[next->backend_id], &head);
                    if (ec != GGML_STATUS_SUCCESS) {
                        worker_join();
                        return ec;
                    }

                    next->prefix_end = prefix;
                } else {
                    MOE_LOG_DEBUG("%s: [CPU_SPLIT] split=%d handed to the worker, nothing independent to"
                                  " launch ahead in split=%d\n", __func__, split_id, join_at);
                }
            }
        }

        // nodes of this split that were already launched as the head of a worker dispatch [TAG_FFN_CPU_SPLIT]
        const int resume_at = split->prefix_end > 0 ? split->prefix_end : 0;

        // [TAG_FFN_CPU_SPLIT] the CPU backend has one scratch context and cannot run two splits at once,
        // so a host split that is not handed to the worker waits for the one that is. In the split FFN
        // graph this never costs anything: the join above already happened at the consumer split.
        if (!on_worker && worker_split >= 0 && ggml_backend_buft_is_host(sched->bufts[split_backend_id])) {
            worker_join();
        }

        if (on_worker) {
            // this split is running on the worker and its result is read by a later split: there is
            // nothing to launch here [TAG_FFN_CPU_SPLIT]
        } else if (!sched->callback_eval) {
            // === INSTRUMENTATION: compute start ===
            MOE_LOG_DEBUG("%s: [COMPUTE] split=%d backend=%d copy_id=%d nodes=%d..%d\n",
                __func__, split_id, split_backend_id, split->copy_id, resume_at, split->graph.n_nodes);
            // ========================================
            enum ggml_status ec;
            if (resume_at > 0) {
                struct ggml_cgraph tail = ggml_graph_view(&split->graph, resume_at, split->graph.n_nodes);
                ec = ggml_backend_graph_compute_async(split_backend, &tail);
            } else {
                ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            }
            if (ec != GGML_STATUS_SUCCESS) {
                worker_join();
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = resume_at; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    worker_join();
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        if (sched->events[split_backend_id][split->copy_id] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][split->copy_id], split_backend);
            // === INSTRUMENTATION: sync events ===
            MOE_LOG_DEBUG("%s: [SYNC] recorded event backend=%d copy_id=%d for split=%d\n",
                __func__, split_backend_id, split->copy_id, split_id);
            // =====================================
        }

        // prefetch the weights of the splits that are still prefetch_window splits ahead now that
        // compute(i) is in flight: each one goes into the slot and cell assigned to that split and
        // overlaps with the compute of the splits in between [weight_prefetch] [GGML_SCHED_PREFETCH_DEPTH]
        //
        // The window slides by one split per iteration and only its newly available right edge is
        // issued, so no weight is ever copied twice. That matters: with weight_prefetch on, a
        // prefetchable weight is skipped by the copy call of its own split (the prefetchable !=
        // weights_only test in copy_split_inputs above), so the prefetch pass is the only place that
        // weight is ever copied. Every split up to n_splits-1 is covered here before its compute runs.
        // With GGML_SCHED_PREFETCH_BURST=1 the very first boundary issues the whole fill (splits
        // 1..depth), which is what makes a depth d window possible without putting d transfers ahead of
        // the first compute of the graph.
        if (sched->weight_prefetch) {
            const int want = std::min(split_id + prefetch_window, sched->n_splits - 1);
            for (int k = prefetched_upto + 1; k <= want; k++) {
                copy_split_inputs(k, true);
            }
            if (want > prefetched_upto) {
                prefetched_upto = want;
            }
            // [GGML_SCHED_H2D_CHUNK] continuation chunks. split k with n chunks gets chunk j at
            // boundary k - n + j: one chunk per boundary over the last n boundaries before its
            // compute, chunk 0 having gone out early at the window edge (or the burst). the window
            // still feeds the copy stream the same total bytes, but no single transfer holds the
            // FIFO longer than one chunk, so the activation copy of the next split never queues
            // behind a whole donor.
            if (sched->h2d_chunk_bytes > 0 || sched->h2d_chunk_auto) {
                for (int k = split_id + 1; k <= want; k++) {
                    const int n = split_chunk_n[k];
                    if (n > 1) {
                        const int j = split_id - (k - n);
                        if (j >= 1 && j < n) {
                            copy_split_inputs(k, true, j);
                        }
                    }
                }
            }
        }


        prev_backend_id = split_backend_id;
        prev_copy_id = split->copy_id;
    }

    // [TAG_FFN_CPU_SPLIT] a host split whose result nothing in this graph reads - the last split of the
    // graph is one - is joined here rather than left running into the next plan, which would have the
    // worker writing into buffers the allocator is about to reuse
    worker_join();

    // [CPU_SPLIT_STATS] one line per 32 graphs, only with GGML_SCHED_CPU_SPLIT_STATS=1
    cpu_split_stats_report();


    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // multiple copy slots are useful for host->device weight copies even without pipeline parallelism
    // [GGML_SCHED_N_COPIES] [GGML_SCHED_PREFETCH]
    // keep the raw strings: the log must be able to tell "not set in this process" apart from "set to 0".
    // this function is the ONLY place where n_copies / weight_prefetch are assigned - nothing later on
    // can lower them, so if the [ALLOC] line reports n_copies=1, the reason is here.
    const char * env_n_copies = getenv("GGML_SCHED_N_COPIES");
    const char * env_prefetch = getenv("GGML_SCHED_PREFETCH");
    static const int  sched_n_copies = env_n_copies ? atoi(env_n_copies) : 0;
    // [fork default] unset now means ON: this build's measured configuration is 2 copy slots with weight
    // prefetch, and the Dense mechanism is what the whole tmp_sched_test measurement series is about.
    // GGML_SCHED_PREFETCH=0 still switches it off, and the [ENV] line below prints "(unset)" next to
    // sched_prefetch=1, so a log can still tell "defaulted on" apart from "asked for".
    static const bool sched_prefetch = env_prefetch == nullptr || atoi(env_prefetch) != 0;

    // [GGML_SCHED_INPUT_SINGLE_COPY] one destination for INPUT-flagged copies, see the field comment
    static const bool sched_input_single_copy = getenv("GGML_SCHED_INPUT_SINGLE_COPY") != nullptr
        && atoi(getenv("GGML_SCHED_INPUT_SINGLE_COPY")) != 0;
    sched->input_single_copy = sched_input_single_copy;

    // === INSTRUMENTATION: static env & n_copies ===
    // INFO level on purpose: this is the single decision point of the 2-slot prefetch pipeline and it
    // has to be visible without enabling the debug log.
    MOE_LOG_INFO("%s: [ENV] GGML_SCHED_PREFETCH=%s GGML_SCHED_N_COPIES=%s parallel=%d -> sched_prefetch=%d sched_n_copies=%d\n",
        __func__, env_prefetch ? env_prefetch : "(unset)", env_n_copies ? env_n_copies : "(unset)",
        parallel, sched_prefetch ? 1 : 0, sched_n_copies);
    // ==============================================

    if (sched_prefetch && sched->n_copies < 2) {
        sched->n_copies = 2;
    }
    if (sched_n_copies > 1) {
        sched->n_copies = std::min(sched_n_copies, GGML_SCHED_MAX_COPIES);
    }
    sched->weight_prefetch = sched_prefetch && sched->n_copies > 1;

    // [GGML_SCHED_H2D_CELLS] number of disjoint H2D destination cells inside each prefetch slot.
    // the H2D copy of split i is issued right after the compute of split i-1 was launched, and the
    // CUDA backend parks the copy stream on the per-slot in_use event of the last split that read
    // that destination. with one cell the destination of split i is the destination of split i-nc,
    // whose compute may still be queued, so the copy stream can park on that event. the reuse distance
    // is lcm(n_copies, n_h2d_cells) splits, so n_h2d_cells = n_copies + 1 used to make it nc*(nc+1).
    // with n_h2d_cells == 1 the slot layout is byte-identical to the previous behaviour.
    //
    // Default is 1 now (it used to be n_copies + 1). Measured on Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S,
    // dense_ab_notes.md 7.19 and 7.21: at 278,2 MiB of host weights per token cells=1 ran at 30,70
    // against 30,80 t/s for cells=3, and at 1239,1 MiB/token the two were identical (18,30 both,
    // C-B = 0,00, with the copy stream busy 92 % of the token in both), while a prefetch slot costs
    // shared_prefix + cells x (largest single weight copy): cells=1 charges 95,62 MiB of staging instead
    // of 278,90. 183,28 MiB of device memory for at most 0,1 t/s is not a trade worth making, so the
    // cheap layout is the default and GGML_SCHED_H2D_CELLS=N opts back into a wider one. Note that with
    // 2 slots and 1 cell the reuse distance is 2 splits, which clamps prefetch_depth to 1: the fixed
    // configuration this default encodes is 2 slots, 1 cell, depth 1, burst 1.
    const char * env_h2d_cells = getenv("GGML_SCHED_H2D_CELLS");
    static const int sched_h2d_cells = env_h2d_cells ? atoi(env_h2d_cells) : 0;
    sched->n_h2d_cells = 1;
    if (sched->weight_prefetch) {
        const int cells = sched_h2d_cells > 0 ? sched_h2d_cells : 1;
        sched->n_h2d_cells = std::min(cells, GGML_SCHED_MAX_H2D_CELLS);
    }

    // [GGML_SCHED_PREFETCH_DEPTH] how many splits ahead the weight copies are issued. 1 is the original
    // one-split lookahead. Unset (or 0) is auto now: pass D resolves it to 1 for a single-cell ring -
    // the value every depth comparison in the notes measured - and to the widest window the reuse
    // distance allows when the caller paid for a wider ring through GGML_SCHED_H2D_CELLS.
    //
    // The useful ceiling is the destination reuse distance minus 1. The destination of split i is reused
    // every lcm(n_copies, n_h2d_cells) splits, so prefetching split i+depth writes a range whose last
    // reader is split i+depth-reuse. At depth >= reuse that reader is a split of the current token whose
    // compute was only just launched, the copy parks on its in_use event, and because there is one copy
    // stream and it is a FIFO, everything queued behind it - including the copies the current token is
    // waiting for - parks too. depth = reuse-1 can therefore never wait on a compute of this token.
    //
    // [GGML_SCHED_PREFETCH_DEPTH] this is the STATIC version of that ceiling: it is computed here, where
    // no graph exists yet, from the period of the (copy_id, cell_id) pair over every split of a backend.
    // That period is the reuse distance of the physical destination range only if every split really
    // copies weights. ggml_backend_sched_reserve (pass D) recomputes it per graph from the splits that
    // actually write a slot range; the value here stays as the fallback and as the ceiling.
    const char * env_prefetch_depth = getenv("GGML_SCHED_PREFETCH_DEPTH");
    // unset or 0 = auto. the window is then chosen per graph in pass D, from the copies that graph
    // actually issues; what is computed here is only the fallback for graphs that never reach pass D.
    const int    sched_prefetch_depth  = env_prefetch_depth ? atoi(env_prefetch_depth) : 0;
    sched->prefetch_depth_auto = sched_prefetch_depth <= 0;
    const int    dest_reuse_distance   = ggml_sched_lcm(sched->n_copies, sched->n_h2d_cells);
    // Because copy_id and cell_id are both taken from the same per-backend split counter, the pair
    // (copy_id, cell_id) only takes lcm(n_copies, n_h2d_cells) values, and cells that no split lands in
    // keep cell_max = 0 and are never allocated. So the knob buys a ring of reuse_distance buffers, not
    // n_copies * n_h2d_cells of them, and a value that shares a factor with n_copies buys nothing:
    // n_copies = 2, n_h2d_cells = 2 leaves the reuse distance at 2. Measured on
    // Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S with 6 ffn_down donors: cells=3 allocated 179,4 MiB of staging
    // against 125,6 MiB for cells=4, because cells=3 spreads the donors over 6 live buffers while cells=4
    // keeps them in 2 of 4 (tmp_sched_test/ffn_cpu_split_notes.md 13.15).
    if (sched->weight_prefetch && sched->n_h2d_cells > 1 && dest_reuse_distance == sched->n_copies) {
        MOE_LOG_INFO("%s: [ENV] GGML_SCHED_H2D_CELLS=%d does not move the reuse distance (%d splits = n_copies): the ring stays as wide as with cells=1, only the cell_max table is wider\n",
            __func__, sched->n_h2d_cells, dest_reuse_distance);
    }
    int prefetch_depth = std::min(std::max(sched_prefetch_depth, 1), GGML_SCHED_MAX_PREFETCH_DEPTH);
    sched->prefetch_depth_req = prefetch_depth;
    prefetch_depth = std::min(prefetch_depth, dest_reuse_distance - 1);
    sched->prefetch_depth = prefetch_depth > 1 ? prefetch_depth : 1;
    // recomputed for every graph in ggml_backend_sched_reserve (pass D); this is the value used until
    // then and the fallback for graphs that never reach pass D [GGML_SCHED_PREFETCH_DEPTH]
    sched->prefetch_depth_eff = sched->prefetch_depth;
    sched->dbg_pw_copies = -1;
    sched->dbg_pw_reuse  = -1;
    sched->dbg_pw_depth  = -1;

    // [GGML_SCHED_PREFETCH_BURST] how many leading splits are prefetched before the first compute of the
    // graph is launched. Unset = prefetch_depth, which is what every depth sweep measured so far. Set it
    // to 1 and the window keeps its depth but its pipeline fill happens at the first split boundary, i.e.
    // after the compute of split 0 is already in flight, so the leading transfers land on compute windows
    // instead of on the idle cycle between two graphs. burst is clamped to the depth: filling more than
    // the window would prefetch splits the window is not allowed to reach.
    const char * env_prefetch_burst = getenv("GGML_SCHED_PREFETCH_BURST");
    const int    sched_prefetch_burst = env_prefetch_burst ? atoi(env_prefetch_burst) : sched->prefetch_depth;
    sched->prefetch_burst = std::min(std::max(sched_prefetch_burst, 1), sched->prefetch_depth);

    // [GGML_SCHED_H2D_CHUNK] the chunk size in MiB. unset or 0 = off; N = that size; "auto" = chunk when
    // the window of this graph is wider than one split, never below GGML_SCHED_H2D_CHUNK_MIN_MIB. the
    // chunk is never smaller than nbytes/depth_eff, so a weight always fits its window: with depth 6 a
    // 37,85 MiB donor cannot be cut below 6,31 MiB no matter what the knob asks for.
    //
    // auto is NOT the default, and a_ct13_c3 of ffn_cpu_split_notes.md 13.58 is the reason: 13 donors of
    // the 27B with cells=3 (auto opened the window, auto cut every donor into 8 MiB chunks) produced
    // draft acceptance 0.00000 - 0 accepted of 995 generated, against 0.70290 with the same prompt in the
    // same model at depth 1. The draft stopped matching the target, which is the same class of failure as
    // 13.51/13.52 (an early reader of a chunked destination ran over bytes no copy had written yet).
    // Until that hole is closed the chunker has to be asked for by name.
    {
        const char * env_h2d_chunk = getenv("GGML_SCHED_H2D_CHUNK_MIB");
        // "auto" is spelled by hand rather than with strcasecmp: this file is compiled by MSVC too, and
        // strcasecmp is not in its <string.h>.
        const std::string h2d_chunk_req = env_h2d_chunk ? std::string(env_h2d_chunk) : std::string();
        sched->h2d_chunk_auto  = !h2d_chunk_req.empty() && std::equal(h2d_chunk_req.begin(), h2d_chunk_req.end(),
                                 "auto", [](const char a, const char b) { return std::tolower((unsigned char) a) == b; });
        sched->h2d_chunk_bytes = (!env_h2d_chunk || sched->h2d_chunk_auto)
                                 ? 0 : (size_t) std::max(0, atoi(env_h2d_chunk)) * 1048576u;
    }

    // [TAG_FFN_CPU_SPLIT] the worker exists if there is a device backend to overlap a host split with.
    // It is on by default and costs nothing when no split is ever marked cpu_async: no thread is
    // created. Since the host-rows mechanism (--ffn-cpu-rows / --ffn-host-rows) was trimmed out of this
    // fork, nothing sets GGML_HINT_SRC0_KEEP_HOST anymore and the worker stays dormant; the skeleton is
    // kept because the donor/-ot host-weight path uses the same split machinery around it.
    // GGML_SCHED_CPU_SPLIT_WORKER=0 is the way back to the serialized behaviour, for A/B against it.
    {
        const char * env_cpu_worker = getenv("GGML_SCHED_CPU_SPLIT_WORKER");
        const bool   cpu_worker_on  = env_cpu_worker ? atoi(env_cpu_worker) != 0 : true;
        sched->cpu_worker = (cpu_worker_on && sched->n_backends > 1) ? new ggml_backend_cpu_split_worker() : NULL;

        MOE_LOG_INFO("%s: [ENV] cpu split worker=%d\n", __func__, sched->cpu_worker ? 1 : 0);
    }

    // === INSTRUMENTATION: final n_copies & weight_prefetch ===
    char depth_desc[32];
    if (sched->prefetch_depth_auto) {
        snprintf(depth_desc, sizeof(depth_desc), "auto(fallback %d)", sched->prefetch_depth);
    } else {
        snprintf(depth_desc, sizeof(depth_desc), "%d", sched->prefetch_depth);
    }
    char chunk_desc[32];
    if (sched->h2d_chunk_auto) {
        snprintf(chunk_desc, sizeof(chunk_desc), "auto(min %d MiB)", GGML_SCHED_H2D_CHUNK_MIN_MIB);
    } else {
        snprintf(chunk_desc, sizeof(chunk_desc), "%d MiB", (int) (sched->h2d_chunk_bytes / 1048576u));
    }
    MOE_LOG_INFO("%s: [ENV] final n_copies=%d, weight_prefetch=%d, n_h2d_cells=%d, dest reuse distance=%d splits, prefetch depth=%s splits, burst=%d splits, h2d chunk=%s\n",
        __func__, sched->n_copies, sched->weight_prefetch ? 1 : 0, sched->n_h2d_cells,
        dest_reuse_distance, depth_desc, sched->prefetch_burst, chunk_desc);
    if (env_prefetch_depth && sched_prefetch_depth > 1 && sched->prefetch_depth != sched_prefetch_depth) {
        MOE_LOG_INFO("%s: [ENV] GGML_SCHED_PREFETCH_DEPTH=%s clamped to %d by the static reuse distance (GGML_SCHED_MAX_PREFETCH_DEPTH=%d, lcm(n_copies, n_h2d_cells)-1=%d). "
                     "this is a ceiling only: the window is recomputed per graph from the real reuse distance, see [PREFETCH_WINDOW]\n",
            __func__, env_prefetch_depth, sched->prefetch_depth, GGML_SCHED_MAX_PREFETCH_DEPTH,
            dest_reuse_distance - 1);
    }
    if (env_prefetch_burst && sched->prefetch_burst != sched_prefetch_burst) {
        MOE_LOG_INFO("%s: [ENV] GGML_SCHED_PREFETCH_BURST=%s clamped to %d (prefetch depth=%d: a fill wider than the window would prefetch splits the window is not allowed to reach)\n",
            __func__, env_prefetch_burst, sched->prefetch_burst, sched->prefetch_depth);
    }
    // ========================================================


    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    // the size of a prefetch slot is the footprint of the largest split that uses it, which is only
    // known after the graph has been split, so the buffers themselves are allocated lazily in
    // ggml_backend_sched_split_graph
    if (sched->weight_prefetch) {
        sched->prefetch_slots     = (ggml_backend_buffer_t *) calloc(sched->n_backends * sched->n_copies, sizeof(ggml_backend_buffer_t));
        sched->prefetch_slot_size = (size_t *)                calloc(sched->n_backends * sched->n_copies, sizeof(size_t));
        GGML_ASSERT(sched->prefetch_slots != NULL);
        GGML_ASSERT(sched->prefetch_slot_size != NULL);

        // tracks the host weight buffers that were already handed to the backend's host-buffer
        // registration (cudaHostRegister), so the async H2D prefetch reads pinned memory. the
        // attempt is remembered whether it succeeded or not to avoid retrying it every token
        sched->pinned_host_bufs_capacity = 256;
        sched->pinned_host_bufs = (void **) calloc(sched->pinned_host_bufs_capacity, sizeof(void *));
        sched->n_pinned_host_bufs = 0;
        sched->pinned_host_ranges = (void **) calloc(sched->pinned_host_bufs_capacity, sizeof(void *));
        sched->n_pinned_host_ranges = 0;
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
        // [TAG_FFN_CPU_SPLIT] created against the device that produced the activation copy, see
        // copy_split_inputs; reused from plan to plan, so it is freed here and nowhere else
        if (sched->splits[i].d2h_event != NULL) {
            ggml_backend_event_free(sched->splits[i].d2h_event);
            sched->splits[i].d2h_event = NULL;
        }
    }
    free(sched->splits);

    // [TAG_FFN_CPU_SPLIT] the destructor stops and joins the thread
    delete sched->cpu_worker;
    sched->cpu_worker = NULL;
    if (sched->prefetch_slots != NULL) {
        for (int i = 0; i < sched->n_backends * sched->n_copies; i++) {
            ggml_backend_buffer_free(sched->prefetch_slots[i]);
        }
        free(sched->prefetch_slots);
        free(sched->prefetch_slot_size);
    }
    if (sched->pinned_host_bufs != NULL) {
        // the buffers themselves belong to the model loader; only the tracking array is ours
        free(sched->pinned_host_bufs);
        sched->pinned_host_bufs = NULL;
        sched->n_pinned_host_bufs = 0;
    }
    if (sched->pinned_host_ranges != NULL) {
        free(sched->pinned_host_ranges);
        sched->pinned_host_ranges = NULL;
        sched->n_pinned_host_ranges = 0;
    }
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    if (!sched->weight_prefetch) {
        sched->next_copy = (sched->next_copy + 1) % sched->n_copies;
    }

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    // === INSTRUMENTATION: alloc_splits debug ===
    ggml_backend_sched_alloc_splits_debug(sched);
    // ===========================================

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    size_t total = ggml_gallocr_get_buffer_size(sched->galloc, backend_index);

    // account for the reusable weight-prefetch slots backing the aliased weight copies,
    // otherwise the VRAM they occupy is invisible to callers tuning -n-gpu-layers [weight_prefetch]
    if (sched->weight_prefetch && sched->prefetch_slots != NULL) {
        const int nc = sched->n_copies;
        for (int c = 0; c < nc; c++) {
            total += sched->prefetch_slot_size[backend_index * nc + c];
        }
    }

    return total;
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

// [TAG_FFN_CPU_SPLIT] a CPU buffer type that is deliberately not the device default: llama-model-load
// views the model file through the ctx of the default buft under --load-mode mmap, and a weight that
// the loader re-quantizes has file bytes that are neither its type nor its size, so it needs real
// memory of its own instead of a file view. Same allocator and host semantics as the CPU buffer type,
// different identity, so the mapping branch (is_default_buft) skips it and it lands in the
// real-allocation branch.
static const char * ggml_backend_cpu_nomapped_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_NoMap";

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_nomapped_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_nomapped_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_nomapped_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_nomapped_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
