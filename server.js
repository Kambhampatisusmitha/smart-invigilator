require('dotenv').config();
const express = require('express');
const nodemailer = require('nodemailer');
const bodyParser = require('body-parser');
const cors = require('cors');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());

// MongoDB Atlas connection
const mongoUri = process.env.MONGO_URI; // Store the Mongo URI in your .env file
mongoose.connect(mongoUri, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB Atlas'))
  .catch((err) => console.error('Failed to connect to MongoDB Atlas:', err));

// Use environment variables for sensitive data
const emailUser = process.env.EMAIL_USER;
const emailPass = process.env.EMAIL_PASS;

let generatedOtp = '';

// Generate a 6-digit OTP
function generateOtp() {
    let otp = '';
    for (let i = 0; i < 6; i++) {
        otp += Math.floor(Math.random() * 10);
    }
    return otp;
}

// MongoDB user schema and model
const userSchema = new mongoose.Schema({
    username: String,
    email: { type: String, unique: true },
    password: String,
});

const User = mongoose.model('User', userSchema);

// Send OTP endpoint
app.post('/send-otp', (req, res) => {
    const { email } = req.body;

    if (!email) {
        return res.status(400).send('Email is required');
    }

    generatedOtp = generateOtp();

    const transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: {
            user: emailUser,
            pass: emailPass,
        },
    });

    const mailOptions = {
        from: emailUser,
        to: email,
        subject: 'Your OTP for Registration',
        text: `Your OTP for registration is: ${generatedOtp}`,
    };

    transporter.sendMail(mailOptions, (error, info) => {
        if (error) {
            console.log(error);
            return res.status(500).send('Error sending OTP');
        }
        console.log(`OTP sent to ${email}: ${generatedOtp}`);
        res.status(200).send('OTP sent successfully');
    });
});

// Verify OTP endpoint
app.post('/verify-otp', (req, res) => {
    const { otp } = req.body;

    if (otp === generatedOtp) {
        return res.status(200).send('OTP verified successfully');
    } else {
        return res.status(400).send('Invalid OTP');
    }
});

// User Login endpoint
app.post('/login', async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).send('Email and password are required');
    }

    try {
        // Look for the user in the database by email
        const user = await User.findOne({ email });

        if (!user) {
            return res.status(404).send('User not found');
        }

        // Compare the provided password with the stored hashed password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(400).send('Invalid password');
        }

        // If credentials are valid, send a success response
        res.status(200).send('Login successful');
    } catch (error) {
        console.error(error);
        res.status(500).send('Server error');
    }
});


// Register user endpoint (Optional: for registering new users)
app.post('/register', async (req, res) => {
    const { username, email, password, otp } = req.body;

    if (otp !== generatedOtp) {
        return res.status(400).send('Invalid OTP');
    }

    try {
        const hashedPassword = await bcrypt.hash(password, 10);

        const newUser = new User({
            username: username,
            email: email,
            password: hashedPassword,
        });

        await newUser.save();
        res.status(201).send('User registered successfully');
    } catch (error) {
        console.error(error);
        res.status(500).send('Error registering user');
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
