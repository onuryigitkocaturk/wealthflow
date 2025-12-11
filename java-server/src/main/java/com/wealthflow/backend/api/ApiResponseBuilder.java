package com.wealthflow.backend.api;

import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;

public class ApiResponseBuilder {

    public static <T> ApiResponse<T> success(HttpServletRequest request, String message, T data) {
        return new ApiResponse<>(
                true,
                message,
                request.getRequestURI(),
                data
        );
    }

    public static <T> ApiResponse<T> success(HttpServletRequest request, T data) {
        return new ApiResponse<>(
                true,
                null,
                request.getRequestURI(),
                data
        );
    }

    public static ApiResponse<?> error(HttpServletRequest request, String message) {
        return new ApiResponse<>(
                false,
                message,
                request.getRequestURI(),
                null
        );
    }
}