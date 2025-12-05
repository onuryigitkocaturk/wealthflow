package com.wealthflow.backend.repository;

import com.wealthflow.backend.model.Allocation;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AllocationRepository extends JpaRepository<Allocation, Long> {
}
