package com.wealthflow.backend.security;

import com.wealthflow.backend.dto.JwtTokenResponse;
import com.wealthflow.backend.model.UserProfile;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;

import java.nio.charset.StandardCharsets;
import java.security.Key;
import java.time.Instant;
import java.util.Date;

@Service
public class JwtService {

    @Value("${jwt.secret}")
    private String secretKey;

    @Value("${jwt.access-token-expiration-ms}")
    private long accessTokenExpirationMs;

    private Key getSigningKey() {
        return Keys.hmacShaKeyFor(secretKey.getBytes(StandardCharsets.UTF_8));
    }

    public JwtTokenResponse generateToken(UserProfile user) {
        String accessToken = generateAccessToken(user);

        return new JwtTokenResponse(
                accessToken,
                accessTokenExpirationMs
        );
    }

    public String generateAccessToken(UserProfile user) {
        Instant now = Instant.now();

        return Jwts.builder()
                .setSubject(user.getEmail())                      // sub
                .setIssuedAt(Date.from(now))                      // iat
                .setExpiration(Date.from(now.plusMillis(accessTokenExpirationMs))) // exp
                .signWith(getSigningKey(), SignatureAlgorithm.HS256)
                .compact();
    }

    public String extractEmail(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(getSigningKey())
                .build()
                .parseClaimsJws(token)
                .getBody()
                .getSubject();
    }

    public boolean isTokenValid(String token, String email) {
        String tokenEmail = extractEmail(token);
        return tokenEmail.equals(email) && !isTokenExpired(token);
    }

    private boolean isTokenExpired(String token) {
        Date exp = Jwts.parserBuilder()
                .setSigningKey(getSigningKey())
                .build()
                .parseClaimsJws(token)
                .getBody()
                .getExpiration();

        return exp.before(new Date());
    }
}
