package com.wealthflow.backend.service;

import com.wealthflow.backend.dto.LoginRequest;
import com.wealthflow.backend.dto.JwtTokenResponse;
import com.wealthflow.backend.dto.RegisterRequest;
import com.wealthflow.backend.model.UserProfile;
import com.wealthflow.backend.repository.UserProfileRepository;
import com.wealthflow.backend.security.JwtService;

import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final AuthenticationManager authenticationManager;
    private final UserProfileRepository userProfileRepository;
    private final JwtService jwtService;
    private final PasswordEncoder passwordEncoder;

    public AuthService(AuthenticationManager authenticationManager,
                       UserProfileRepository userProfileRepository,
                       JwtService jwtService,
                       PasswordEncoder passwordEncoder) {

        this.authenticationManager = authenticationManager;
        this.userProfileRepository = userProfileRepository;
        this.jwtService = jwtService;
        this.passwordEncoder = passwordEncoder;
    }

    public JwtTokenResponse login(LoginRequest request) {

        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        request.email(),
                        request.password()
                )
        );

        UserProfile user = userProfileRepository
                .findByEmail(request.email())
                .orElseThrow(() -> new RuntimeException("User not found"));
        return jwtService.generateToken(user);
    }
    public JwtTokenResponse register(RegisterRequest request) {

        if (userProfileRepository.findByEmail(request.email()).isPresent()) {
            throw new RuntimeException("Email already registered");
        }

        UserProfile user = new UserProfile();
        user.setEmail(request.email());
        user.setName(request.fullName());
        user.setPasswordHash(passwordEncoder.encode(request.password()));

        userProfileRepository.save(user);

        return jwtService.generateToken(user);
    }

}
